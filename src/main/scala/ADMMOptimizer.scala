package com.tulloch.admmlrspark

import ADMMOptimizer._
import DenseVectorImplicits._
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.Logging
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import scala.math.{abs, exp, log, log1p, max, min}

/**
 * The state kept on each partition - the data points, and the x,
 * y, u vectors at each iteration
 */
case class ADMMState(
  val points: Array[LabeledPoint],
  val x: Vector,
  val z: Vector,
  val u: Vector)
    extends Logging {

  def objective(rho: Double)(weights: Vector): Double = {
    val lossObjective = points
      .map(lp => {
        val margin = lp.label * (weights dot Vector(lp.features))
        -logPhi(margin)
      })
      .sum

    val regularizerObjective = (weights - z + u).squaredNorm
    val totalObjective = lossObjective + rho / 2 * regularizerObjective
    totalObjective
  }

  def gradient(rho: Double)(weights: Vector): Vector = {
    val lossGradient = points
      .map(lp => {
        val margin = lp.label * (weights dot Vector(lp.features))
        lp.label * Vector(lp.features) * (phi(margin) - 1)
      })
      .reduce(_ + _)

    val regularizerGradient = 2 * (weights - z + u)
    val totalGradient = lossGradient + rho / 2 * regularizerGradient
    totalGradient
  }

  private def clampToRange(lower: Double, upper: Double)(margin: Double): Double =
    min(upper, max(lower, margin))

  private def logPhi(margin: Double): Double = {
    val t = clampToRange(-10, 10)(margin)
    math.log(1.0 / (1.0 + exp(-t)))
    // if (t > 0) - math.log(1.0 + exp(-t)) else t - math.log(1 + exp(t)) 
  }

  private def phi(margin: Double): Double = {
    val t = clampToRange(-10, 10)(margin)
    if (t > 0) 1.0 / (1 + exp(-t)) else exp(t) / (1 + exp(t))
  }
}

object ADMMState {
  def apply(points: Array[LabeledPoint]): ADMMState = {
    val numFeatures = points.head.features.length
    new ADMMState(
      points = points,
      x = zeroes(numFeatures),
      z = zeroes(numFeatures),
      u = zeroes(numFeatures)
    )
  }
}

class ADMMOptimizer(
  val numIterations: Int,
  val lambda: Double,
  val rho: Double)
    extends Optimizer with Logging with Serializable {

  override def optimize(
    data: RDD[(Double, Array[Double])],
    initialWeights: Array[Double]): Array[Double] = {
    val numPartitions = data.partitions.length
    // Hash each datapoint to a partition

    // The input points are 0,1 - we map to (-1, 1) for consistency
    // with the presentation in
    // http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
    val admmStates = data
      .map(p => {
        val scaledLabel = 2 * p._1 - 1
        new LabeledPoint(scaledLabel, p._2)
      })
      .groupBy(lp => {
        lp.hashCode() % numPartitions
      })
      .map(ip => ADMMState(ip._2.toArray))
     
    // Run numIterations of runRound
    val finalStates = (1 to numIterations).foldLeft(admmStates)((s, _) => runRound(s))

    // return average of final weight vectors across the partitions
    average(finalStates.map(_.x)).elements
  }

  private def runRound(states: RDD[ADMMState]): RDD[ADMMState] =
    (xUpdate _ andThen zUpdate _ andThen uUpdate _)(states)

  private def xUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(parallelXUpdate)

  private def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] = {
    val numPartitions = states.partitions.length
    val epsilon = 0.00001 // avoid division by zero for shrinkage
    val xBar = average(states.map(_.x))
    val uBar = average(states.map(_.u))
    val zNew = Vector((xBar + uBar)
      .elements
      .map(shrinkage(lambda / (rho * numPartitions + epsilon))))

    states.map(state => state.copy(z = zNew))
  }

  private def uUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(state => state.copy(u = state.u + state.x - state.z))

  private def parallelXUpdate(state: ADMMState): ADMMState = {
    // Our convex objective function that we seek to optimize
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]) = {
        val objective = state.objective(rho)(x)
        val gradient = state.gradient(rho)(x)
        (objective, gradient)
      }
    }

    // TODO(tulloch) - it would be nice to have relative tolerance and
    // absolute tolerance here.
    val xNew = new LBFGS[DenseVector[Double]](maxIter=5, m=10).minimize(
      f,
      state.x // this is the "warm start" approach
    )
    state.copy(x = xNew)
  }

  // Given an RDD list of vectors, computes the component-wise average vector.
  def average(updates: RDD[Vector]): Vector = {
    updates.reduce(_ + _) / updates.count
  }
}

object ADMMOptimizer {
  // Eq (4.2) in http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
  def shrinkage(kappa: Double)(v: Double) =
    max(0, v - kappa) - max(0, -v - kappa)

  def zeroes(n: Int) = {
    Vector(Array.fill(n){ 0.0 })
  }
}

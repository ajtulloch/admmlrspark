package com.tulloch.admmlrspark

import DenseVectorImplicits._
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.Logging
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import scala.math.{abs, exp, log1p, max, pow}

class ADMMOptimizer(
  val numFeatures: Int,
  val numIterations: Int,
  val lambda: Double,
  val rho: Double)
    extends Optimizer with Logging {

  /**
   * The state kept on each partition - the data points, and the x,
   * y, u vectors at each iteration
   */
  case class ADMMState(points: Array[LabeledPoint], x: Vector, z: Vector, u: Vector)

  def optimize(
    data: RDD[(Double, Array[Double])],
    initialWeights: Array[Double]): Array[Double] = {
    // Hash each datapoint to a partition 
    val admmStates = data
      .map(p => LabeledPoint(p._1, p._2))
      .groupBy(lp => lp.hashCode % data.partitions.length)
      .map(ip => ADMMState(ip._2.toArray, prior, prior, prior))
      .cache() // TODO(tulloch) - does this do anything?

    // Run numIterations of runRound
    val finalStates = (1 to numIterations).foldLeft(admmStates)((s, _) => runRound(s))

    // return average of final weight vectors across the partitions
    average(finalStates.map(_.x)).elements
  }

  private def runRound(states: RDD[ADMMState]): RDD[ADMMState] =
    (xUpdate _ andThen zUpdate _ andThen uUpdate _)(states)

  private def parallelXUpdate(state: ADMMState): ADMMState = {
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]) = {
        val lossFunction = (x: Vector) => {
          val loss = state.points
            .map(lp => log1p(exp(-lp.label * (x dot Vector(lp.features)))))
            .sum
          val regularizer = squaredNorm(x - state.z + state.u)
          loss + rho / 2 * regularizer
        }

        val gradientFunction = (x: Vector) => {
          val logit = (v: Double) => 1 + exp(-v)
          val lossGradient = state.points
            .map(lp => {
              lp.label *
              Vector(lp.features) *
              (logit(lp.label * (x dot Vector(lp.features))) - 1)
            })
            .reduce(_ + _)
          val regularizerGradient = 2 * (x - state.z + state.u)
          lossGradient + rho / 2 * regularizerGradient
        }
        (lossFunction(x), gradientFunction(x))
      }
    }

    val xNew = new LBFGS[DenseVector[Double]].minimize(f, state.x)

    ADMMState(state.points, xNew, state.z, state.u)
  }

  private def squaredNorm(v: Vector): Double = v.elements.map(pow(_, 2)).sum

  private def xUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(parallelXUpdate)

  private def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] = {
    val xBar = average(states.map(_.x))
    val uBar = average(states.map(_.u))
    val zNew = Vector((xBar + uBar)
      .elements
      .map(shrinkage(lambda / rho * numFeatures)))
    states.map(s => ADMMState(s.points, s.x, zNew, s.u))
  }

  private def uUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(s => ADMMState(s.points, s.x, s.z, s.u + s.x - s.z))

  private def shrinkage(kappa: Double)(v: Double) =
    if (v != 0) max(0, (1 - kappa / abs(v))) * v else 0

  private def average(updates: RDD[Vector]): Vector =
    updates.reduce(_ + _) / updates.count

  private def prior = Vector(new Array[Double](numFeatures))
}

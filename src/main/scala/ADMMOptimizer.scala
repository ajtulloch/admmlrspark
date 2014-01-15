package com.tulloch.admmlrspark

import ADMMOptimizer._
import DenseVectorImplicits._
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.Logging
import org.apache.spark.SparkContext
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import scala.math.{abs, exp, log, log1p, max, min, pow}

case class SVMADMMPrimalUpdater(
  rho: Double,
  cee: Double,
  lbfgsMaxNumIterations: Int = 5,
  lbfgsHistory: Int = 10,
  lbfgsTolerance: Double = 1E-4) extends ADMMUpdater {

  def xUpdate(state: ADMMState): ADMMState = {
    // Our convex objective function that we seek to optimize
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]) = {
        (objective(state)(x), gradient(state)(x))
      }
    }

    val lbfgs = new LBFGS[DenseVector[Double]](
      maxIter = lbfgsMaxNumIterations,
      m = lbfgsHistory,
      tolerance = lbfgsTolerance)

    val xNew = lbfgs.minimize(f, state.x) // this is the "warm start" approach
    state.copy(x = xNew)
  }

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] = {
    val numerator = states.map(state => state.x + state.u).reduce(_ + _)
    val denominator = states.count + 1.0 / rho
    val newZ = numerator / denominator
    states.map(_.copy(z = newZ))
  }

  def objective(state: ADMMState)(x: Vector): Double = {
    // Eq (12) in
    // http:web.eecs.umich.edu/~honglak/aistats12-admmDistributedSVM.pdf
    val v = state.z - state.u
    val regularizer = (x - v).squaredNorm
    val loss = state.points
      .map{ case LabeledPoint(label, features) => {
        pow(max(1.0 - label * (x dot Vector(features)), 0), 2)
      }}
      .sum

    cee * loss + rho / 2 * regularizer
  }

  def gradient(state: ADMMState)(x: Vector): Vector = {
    // Eq (20) in
    // http:web.eecs.umich.edu/~honglak/aistats12-admmDistributedSVM.pdf
    val v = state.z - state.u
    val regularizer = x - v

    val loss = state.points.map{ case LabeledPoint(label, features) => {
      val margin = max(1.0 - label * (x dot Vector(features)), 0)
      if (margin <= 0) {
        ADMMState.zeroes(x.length)
      } else {
        Vector(features) * (x dot Vector(features)) - label * Vector(features)
      }
    }}
    .reduce(_ + _)

    rho * regularizer - 2 * cee * loss
  }
}

case class LassoRegressionADMMUpdater(
  rho: Double,
  lambda: Double,
  sc: SparkContext,
  ridgeNumPartitions: Int,
  ridgeNumIterations: Int) extends ADMMUpdater {
  def xUpdate(state: ADMMState): ADMMState = {
    // Section (8.2.1) from
    //http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

    // ||Ax - b||^2 + lambda ||x - x_0||^2
    // let z = x - x_0
    // ||A(z + x_0) - b||^2 + lambda ||z||^2
    // ||A(z) + (b - Ax_0)||^2 = lambda ||z||^2
    // -> z = (A^T A + lambdaI)^-1 (A^T (b - Ax_0))
    // -> x = x_0 + (A^T A + lambdaI)^-1 (A^T (b - Ax_0))
    
    val x_0 = state.z - state.u
    val adjustedData = state.points.map{case LabeledPoint(label, features) =>
      // rescale label to recenter the features about Tikhonov prior.
      val newLabel = label - (Vector(features) dot x_0)
      // remove intercept (this is added by the RidgeRegression routine)
      val newFeatures = features.tail
      LabeledPoint(newLabel, features.tail)
    }

    val ridgeRegressionData = sc.makeRDD(state.points, ridgeNumPartitions)
    val ridgeSolution = RidgeRegressionWithSGD.train(ridgeRegressionData, ridgeNumIterations)
    state.copy(x = x_0 + Vector(ridgeSolution.weights))
  }

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    ADMMUpdater.linearZUpdate(lambda = lambda, rho = rho)(states)
}

class ADMMOptimizer(
  val numIterations: Int,
  val updater: ADMMUpdater)
    extends Optimizer with Logging with Serializable {

  override def optimize(
    data: RDD[(Double, Array[Double])],
    initialWeights: Array[Double]): Array[Double] = {
    val numPartitions = data.partitions.length

    val admmStates = data
      .map{case(zeroOnelabel, features) => {
        // The input points are 0,1 - we map to (-1, 1) for consistency
        // with the presentation in
        // http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
        val scaledLabel = 2 * zeroOnelabel - 1
        new LabeledPoint(scaledLabel, features)
      }}

      .groupBy{lp => {
        // map each data point to a given ADMM partition
        lp.hashCode() % numPartitions
      }}
      .map{case (_, points) => ADMMState(points, initialWeights) }

    // Run numIterations of runRound
    val finalStates = (1 to numIterations)
      .foldLeft(admmStates)((s, _) => runRound(s))

    // return average of final weight vectors across the partitions
    average(finalStates.map(_.x)).elements
  }

  private def runRound(states: RDD[ADMMState]): RDD[ADMMState] =
    // run the updates sequentially. Note that the xUpdate and uUpdate
    // happen in parallel, while the zUpdate collects the xUpdates
    // from the mappers.
    (xUpdate _ andThen zUpdate _ andThen uUpdate _)(states)

  private def xUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(updater.xUpdate)

  private def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    updater.zUpdate(states)

  private def uUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    states.map(updater.uUpdate)
}

object ADMMOptimizer {
  // Eq (4.2) in http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

  // Given an RDD list of vectors, computes the component-wise average vector.
  def average(updates: RDD[Vector]): Vector = {
    updates.reduce(_ + _) / updates.count
  }
}

package com.tulloch.admmlrspark

import DenseVectorImplicits._
import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector

case class SVMADMMUpdater(
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

  def objective(state: ADMMState)(weights: Vector): Double = {
    // Eq (12) in
    // http:web.eecs.umich.edu/~honglak/aistats12-admmDistributedSVM.pdf
    val v = state.z - state.u
    val regularizer = (weights - v).squaredNorm
    val loss = state.points
      .map{ case LabeledPoint(label, features) => {
        math.pow(math.max(1.0 - label * (weights dot Vector(features)), 0), 2)
      }}
      .sum

    cee * loss + rho / 2 * regularizer
  }

  def gradient(state: ADMMState)(weights: Vector): Vector = {
    // Eq (20) in
    // http:web.eecs.umich.edu/~honglak/aistats12-admmDistributedSVM.pdf
    val v = state.z - state.u
    val regularizer = weights - v

    val loss = state.points.map{ case LabeledPoint(label, features) => {
      val margin = math.max(1.0 - label * (weights dot Vector(features)), 0)
      if (margin <= 0) {
        ADMMState.zeroes(weights.length)
      } else {
        // \sum (x x^T) * weights
        (Vector(features) * (weights dot Vector(features))) - label * Vector(features)
      }
    }}
    .reduce(_ + _)

    rho * regularizer - 2 * cee * loss
  }
}

class SVMWithADMM(
  val numIterations: Int,
  val rho: Double,
  val cee: Double)
    extends GeneralizedLinearAlgorithm[SVMModel]
    with Serializable {

  override val optimizer = new ADMMOptimizer(
    numIterations,
    new SVMADMMUpdater(rho = rho, cee = cee))

  override val validators = List(DataValidators.classificationLabels)

  override def createModel(
    weights: Array[Double],
    intercept: Double): SVMModel = new SVMModel(weights, intercept)
}

object SVMWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    rho: Double,
    cee: Double) = {
    new SVMWithADMM(numIterations, rho, cee).run(input)
  }
}

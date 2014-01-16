package com.tulloch.admmlrspark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LassoModel
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector
import org.jblas.DoubleMatrix

case class LassoRegressionADMMUpdater(
  lambda: Double,
  rho: Double,
  sc: SparkContext,
  ridgeNumIterations: Int,
  ridgeNumPartitions: Int,
  ridgeStepSize: Double) extends ADMMUpdater {
  def xUpdate(state: ADMMState): ADMMState = {
    // Section (8.2.1) from
    //http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf

    // ||Ax - b||^2 + lambda ||x - x_0||^2
    // let z = x - x_0
    // ||A(z + x_0) - b||^2 + lambda ||z||^2
    // ||A(z) + (b - Ax_0)||^2 = lambda ||z||^2
    // -> z = (A^T A + lambda I)^-1 (A^T (b - Ax_0))
    // -> x = x_0 + (A^T A + lambda I)^-1 (A^T (b - Ax_0))
    
    val x_0 = state.z - state.u
    val adjustedData = state.points.map{case LabeledPoint(label, features) =>
      // rescale label to recenter the features about Tikhonov prior.
      val newLabel = label - (Vector(features) dot x_0)
      // RidgeRegressionWithSGD adds an intercept term for us, so
      //remove input feature with the intercept.
      LabeledPoint(newLabel, features.tail)
    }

    val ridgeRegressionData = sc.makeRDD(state.points, ridgeNumPartitions)
    val ridgeSolution =
      RidgeRegressionWithSGD.train(
        input = ridgeRegressionData,
        stepSize = ridgeStepSize,
        numIterations = ridgeNumIterations,
        regParam = rho)
    state.copy(x = x_0 + Vector(ridgeSolution.weights))
  }

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] =
    ADMMUpdater.linearZUpdate(lambda = lambda, rho = rho)(states)
}

class LassoRegressionWithADMM(
  numIterations: Int,
  lambda: Double,
  rho: Double,
  sc: SparkContext,
  ridgeNumIterations: Int,
  ridgeNumPartitions: Int,
  ridgeStepSize: Int)
    extends GeneralizedLinearAlgorithm[LassoModel]
    with Serializable {

  override val optimizer = new ADMMOptimizer(
    numIterations,
    new LassoRegressionADMMUpdater(lambda, rho, sc, ridgeNumIterations, ridgeNumPartitions, ridgeStepSize))

  def createModel(weights: Array[Double], intercept: Double) =
    new LassoModel(weights, intercept)
}

object LassoRegressionWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    lambda: Double,
    rho: Double,
    ridgeNumIterations: Int,
    ridgeNumPartitions: Int,
    ridgeStepSize: Int) =
    new LassoRegressionWithADMM(
      numIterations, lambda, rho, input.sparkContext, ridgeNumIterations, ridgeNumPartitions, ridgeStepSize).run(input)
}

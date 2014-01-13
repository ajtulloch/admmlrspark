package com.tulloch.admmlrspark

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class LogisticRegressionWithADMM(
  val numFeatures: Int,
  val numIterations: Int,
  val lambda: Double,
  val rho: Double)
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel] {

  override val optimizer = new ADMMOptimizer(numFeatures, numIterations, lambda, rho)

  override def createModel(
    weights: Array[Double],
    intercept: Double): LogisticRegressionModel = {
    new LogisticRegressionModel(weights, intercept)
  }
}

object LogisticRegressionWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    lambda: Double,
    rho: Double): LogisticRegressionModel = {
    val numFeatures = input.first.features.length

    new LogisticRegressionWithADMM(numFeatures, numIterations, lambda, rho).run(input)
  }
}

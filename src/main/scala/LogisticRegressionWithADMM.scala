package com.tulloch.admmlrspark

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

class SparseLogisticRegressionWithADMM(
  val numIterations: Int,
  val lambda: Double,
  val rho: Double)
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel]
    with Serializable {

  override val optimizer = new ADMMOptimizer(
    numIterations,
    lambda,
    rho,
    new SparseLogisticRegressionADMMPrimalUpdater(rho, lambda))

  override val validators = List(DataValidators.classificationLabels)

  override def createModel(
    weights: Array[Double],
    intercept: Double): LogisticRegressionModel = {
    new LogisticRegressionModel(weights, intercept)
  }
}

object SparseLogisticRegressionWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    lambda: Double,
    rho: Double) = {
    new SparseLogisticRegressionWithADMM(numIterations, lambda, rho).run(input)
  }
}

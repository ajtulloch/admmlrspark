package com.tulloch.admmlrspark

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.util.MLUtils

object Main {
  def main(args: Array[String]) {
    if (args.length != 5) {
      println("Usage: LogisticRegression <master> <input_dir> <niters> <lambda> <rho>")
      System.exit(1)
    }
    
    val sc = new SparkContext(args(0), "ADMMLogisticRegression")
    val data = MLUtils.loadLabeledData(sc, args(1))
    val model = SparseLogisticRegressionWithADMM.train(
      data,
      args(2).toInt,
      args(3).toDouble,
      args(4).toDouble)

    sc.stop()
  }
}

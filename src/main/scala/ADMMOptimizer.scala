package com.tulloch.admmlrspark

import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class ADMMOptimizer(
  val numIterations: Int,
  val updater: ADMMUpdater)
    extends Optimizer with Serializable {

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
    ADMMUpdater.average(finalStates.map(_.x)).elements
  }

  private def runRound(states: RDD[ADMMState]): RDD[ADMMState] = {
    // run the updates sequentially. Note that the xUpdate and uUpdate
    // happen in parallel, while the zUpdate collects the xUpdates
    // from the mappers.
    val xUpdate = (s: RDD[ADMMState]) => s.map(updater.xUpdate)
    val zUpdate = (s: RDD[ADMMState]) => updater.zUpdate(s)
    val uUpdate = (s: RDD[ADMMState]) => s.map(updater.uUpdate)

    (xUpdate andThen zUpdate andThen uUpdate)(states)
  }
}



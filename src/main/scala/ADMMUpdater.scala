package com.tulloch.admmlrspark

import org.apache.spark.rdd.RDD

trait ADMMUpdater {
  def xUpdate(state: ADMMState): ADMMState

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState]

  def uUpdate(state: ADMMState): ADMMState = {
    state.copy(u = state.u + state.x - state.z)
  }
}


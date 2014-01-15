package com.tulloch.admmlrspark

import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector

trait ADMMUpdater {
  def xUpdate(state: ADMMState): ADMMState

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState]

  def uUpdate(state: ADMMState): ADMMState = {
    state.copy(u = state.u + state.x - state.z)
  }
}

object ADMMUpdater {
  def linearZUpdate(lambda: Double, rho: Double)(states: RDD[ADMMState]): RDD[ADMMState] = {
    val numStates = states.count
    // TODO(tulloch) - is this epsilon > 0 a hack?
    val epsilon = 0.00001 // avoid division by zero for shrinkage

    // TODO(tulloch) - make sure this only sends x, u to the reducer
    // instead of the full ADMM state.
    val xBar = average(states.map(_.x))
    val uBar = average(states.map(_.u))

    val zNew = Vector((xBar + uBar)
      .elements
      .zipWithIndex
      .map{case(el, index) => {
        // Eq (3) in
        // http://intentmedia.github.io/assets/2013-10-09-presenting-at-ieee-big-data/pld_js_ieee_bigdata_2013_admm.pdf
        // Don't regularize the intercept term
        if (index == 0) el else shrinkage(lambda / (rho * numStates + epsilon))(el)
      }})

    states.map(state => state.copy(z = zNew))
  }

  def shrinkage(kappa: Double)(v: Double) =
    math.max(0, v - kappa) - math.max(0, -v - kappa)

  def average(updates: RDD[Vector]): Vector = {
    updates.reduce(_ + _) / updates.count
  }
}



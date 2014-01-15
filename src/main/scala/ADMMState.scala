package com.tulloch.admmlrspark

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Vector
import scala.math.{abs, exp, log, log1p, max, min}
import DenseVectorImplicits._
import org.apache.spark.Logging

/**
 * The state kept on each partition - the data points, and the x,
 * y, u vectors at each iteration.
 */
case class ADMMState(points: Array[LabeledPoint], x: Vector, z: Vector, u: Vector)

object ADMMState {
  def apply(points: Seq[LabeledPoint], initialWeights: Array[Double]): ADMMState = {
    new ADMMState(
      points = points.toArray,
      x = Vector(initialWeights),
      z = zeroes(initialWeights.length),
      u = zeroes(initialWeights.length)
    )
  }

  def zeroes(n: Int) = {
    Vector(Array.fill(n){ 0.0 })
  }
}

package com.tulloch.admmlrspark

import breeze.linalg.DenseVector
import org.apache.spark.util.Vector
import scala.math.pow

object DenseVectorImplicits {
  implicit def denseVector2Vector(value: DenseVector[Double]): Vector = {
    Vector(value.data)
  }

  implicit def vector2DenseVector(value: Vector): DenseVector[Double] = {
    DenseVector[Double](value.elements)
  }

  class RichVector(v: Vector) {
    def squaredNorm: Double = v.elements.map(pow(_, 2)).sum
  }

  implicit def richVector(v: Vector) = new RichVector(v)
}


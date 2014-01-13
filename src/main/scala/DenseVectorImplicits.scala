package com.tulloch.admmlrspark

import breeze.linalg.DenseVector 
import org.apache.spark.util.Vector

object DenseVectorImplicits {
   implicit def denseVector2Vector(value: DenseVector[Double]): Vector = {
     Vector(value.data)
   }

  implicit def vector2DenseVector(value: Vector): DenseVector[Double] = {
    DenseVector[Double](value.elements)
  }
}

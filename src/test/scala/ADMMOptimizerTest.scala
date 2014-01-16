package com.tulloch.admmlrspark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Vector
import scala.util.Random
import scala.math.{max, abs}
import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

class ADMMSpecification extends PropSpec with GeneratorDrivenPropertyChecks {
  val positiveInts = for (n <- Gen.choose(1, 100)) yield n
  val shrinkageParams = for {
    kappa <- Gen.choose(0.0, 5.0)
    v <- Gen.choose(-100.0, 100.0)
  } yield (kappa, v)

  property("zeroes is implemented correctly") {
    forAll (positiveInts) { (n: Int) =>
      val x = ADMMState.zeroes(n)
      x.length == n && x.elements.min == 0 && x.elements.max == 0
    }
  }

  property("shrinkage is implemented correctly") {
    forAll (shrinkageParams) { (kv: (Double, Double)) =>
      val (kappa, v) = kv
      val calculated =
        if (v > kappa) v - kappa else if (abs(v) <= kappa) 0 else v + kappa

      ADMMUpdater.shrinkage(kappa)(v) == calculated
    }
  }
}

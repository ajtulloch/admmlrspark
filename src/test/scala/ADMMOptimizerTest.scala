package com.tulloch.admmlrspark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import scala.util.Random
import scala.math.abs
import org.scalatest._
import prop._
import matchers._

class LogisticRegressionWithADMMSpecification
    extends PropSpec
    with BeforeAndAfterAll
    with ShouldMatchers
    with GeneratorDrivenPropertyChecks {

  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }


  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  // Generate input of the form Y = logistic(offset + scale*X)
  def generateLogisticInput(
      offset: Double,
      scale: Double,
      nPoints: Int,
      seed: Int): Seq[LabeledPoint]  = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    // NOTE: if U is uniform[0, 1] then ln(u) - ln(1-u) is Logistic(0,1)
    val unifRand = new Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0-u)
    }

    // y <- A + B*x + rLogis()
    // y <- as.numeric(y > 0)
    val y: Seq[Int] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1 else 0
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), Array(x1(i))))
    testData
  }

  def near(eps: Double)(actual: Double, expected: Double): Boolean =
    abs(actual - expected) < eps

  property("recovers A + BX") {
    forAll("a", "b") { (a: Double, b: Double) =>
      val nPoints = 10000
      val numIterations = 10
      val lambda = 0.0
      val rho = 0.0
      val tolerance = 0.1

      val testData = generateLogisticInput(a, b, nPoints, 42)

      val testRDD = sc.parallelize(testData, 2)
      testRDD.cache()
      val lr = new LogisticRegressionWithADMM(numIterations, lambda, rho)

      val model = lr.run(testRDD)

      // Test the weights

      val close = near(tolerance) _

      close(model.intercept, a) && close(model.weights(0), b)
    }
  }
}



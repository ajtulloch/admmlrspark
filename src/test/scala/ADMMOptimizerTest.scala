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
      val x = ADMMOptimizer.zeroes(n)
      x.length == n && x.elements.min == 0 && x.elements.max == 0
    }
  }

  property("shrinkage is implemented correctly") {
    forAll (shrinkageParams) { (pair: (Double, Double)) =>
      val (kappa, v) = pair
      val calculated = if (v > kappa)
        v - kappa
      else if (abs(v) <= kappa)
        0
      else
        v + kappa
      ADMMOptimizer.shrinkage(kappa)(v) == calculated
    }
  }
}

class LogisticRegressionWithADMMCases extends FunSuite with ShouldMatchers {
  test("gradient correct") {
    val state = ADMMState(
      points = Array(
        LabeledPoint(label = 1.0, features = Array(1.0))
      ),
      x = Vector(0.0),
      z = Vector(0.0),
      u = Vector(0.0)
    )
    val stepSize = 0.05
    val weights = Vector(1.0)

    val objective = state.objective(0.0)(weights)
    val grad = state.gradient(0.0)(weights)

    val newWeights = stepSize * grad + weights
    val newObjective = state.objective(0.0)(newWeights)

    // gradient points in the right direction
    newObjective should be > objective
    // function is convex
    newObjective should be > (objective + grad(0) * stepSize)
  }
}

class LogisticRegressionWithADMMSpecification
    extends PropSpec
    with BeforeAndAfterAll
    with ShouldMatchers
    with TableDrivenPropertyChecks
    with GeneratorDrivenPropertyChecks {

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

  var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  property("recovers A + BX") {
    forAll(Table(
      ("a", "b"),
      (0.0, 0.0),
      (1.0, -1.0),
      (0.05, 1.05),
      (3.0, 3.0)
    )) { (a: Double, b: Double) =>
      val nPoints = 10000
      val numIterations = 3
      val lambda = 0.0
      val rho = 0.000

      val testData = generateLogisticInput(a, b, nPoints, 42)

      val testRDD = sc.parallelize(testData, 5)
      testRDD.cache()
      val lr = new LogisticRegressionWithADMM(numIterations, lambda, rho)

      val model = lr.run(testRDD)

      // Test the weights
      model.intercept should be (a plusOrMinus (max(0.03, a * 0.05)))
      model.weights(0) should be (b plusOrMinus (max(0.03, b * 0.05)))
    }
  }
}



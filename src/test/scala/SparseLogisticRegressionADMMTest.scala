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

class SparseLogisticRegressionWithADMMCases extends FunSuite with ShouldMatchers {
  test("gradient correct") {
    val state = ADMMState(
      points = Array(LabeledPoint(label = 1.0, features = Array(1.0))),
      initialWeights = Array(0.0)
    )

    val updater = SparseLogisticRegressionADMMPrimalUpdater(rho = 0.0, lambda = 0.0)

    val stepSize = 0.05
    val weights = Vector(1.0)

    val objective = updater.objective(state)(weights)
    val grad = updater.gradient(state)(weights)

    val newWeights = stepSize * grad + weights
    val newObjective = updater.objective(state)(newWeights)

    // gradient points in the right direction
    newObjective should be > objective
    // objective function is convex.
    newObjective should be > (objective + grad(0) * stepSize)
  }
}

class SparseLogisticRegressionWithADMMSpecification
    extends PropSpec
    with BeforeAndAfterAll
    with ShouldMatchers
    with TableDrivenPropertyChecks
    with GeneratorDrivenPropertyChecks {

  override def beforeAll() {
    sc = new SparkContext("local[8]", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  // Generate input of the form Y = logistic(offset + scale*X)
  // Copypasta'd from the Spark LogisticRegressionSuite.scala
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

  @transient private var sc: SparkContext = _

  property("recovers A + BX") {
    forAll(Table(
      ("a", "b"),
      (0.0, 0.0),
      (1.0, -1.0),
      (0.05, 1.05),
      (3.0, 3.0)
    )) { (a: Double, b: Double) =>
      val nPoints = 10000
      val numIterations = 5
      val lambda = 0.0
      val rho = 0.000

      val testData = generateLogisticInput(a, b, nPoints, 42)

      val testRDD = sc.parallelize(testData, 5)
      testRDD.cache()
      val lr = new SparseLogisticRegressionWithADMM(numIterations, lambda, rho)

      val model = lr.run(testRDD)

      val near = (v: Double) => v plusOrMinus(max(0.05, v * 0.05))
      model.intercept should be (near(a))
      model.weights(0) should be (near(b))
    }
  }
}



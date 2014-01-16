package com.tulloch.admmlrspark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Vector
import org.jblas.DoubleMatrix
import scala.util.Random
import scala.math.{max, abs}
import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

class SVMWithADMMCases extends FunSuite with ShouldMatchers {
  test("gradient correct") {
    val state = ADMMState(
      points = Array(LabeledPoint(label = 1.0, features = Array(1.0))),
      initialWeights = Array(0.0)
    )

    val updater = SVMADMMUpdater(rho = 0.0, cee = 1.0)

    val stepSize = 0.05
    val weights = Vector(0.5)

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

class SVMADMMSpecification
    extends PropSpec
    with BeforeAndAfterAll
    with ShouldMatchers
    with TableDrivenPropertyChecks
    with GeneratorDrivenPropertyChecks {

  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local[8]", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  def generateSVMInput(
    intercept: Double,
    weights: Array[Double],
    nPoints: Int,
    seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val weightsMat = new DoubleMatrix(1, weights.length, weights:_*)
    val x = Array.fill[Array[Double]](nPoints)(
        Array.fill[Double](weights.length)(rnd.nextDouble() * 2.0 - 1.0))
    val y = x.map { xi =>
      val yD = (new DoubleMatrix(1, xi.length, xi:_*)).dot(weightsMat) +
        intercept + 0.01 * rnd.nextGaussian()
      if (yD < 0) 0.0 else 1.0
    }
    y.zip(x).map(p => LabeledPoint(p._1, p._2))
  }

  def validatePrediction(predictions: Seq[Double], input: Seq[LabeledPoint]): Double = {
    val numOffPredictions = predictions.zip(input).filter { case (prediction, expected) =>
      (prediction != expected.label)
    }.size

    numOffPredictions / input.length
  }

  property("recovers A + BX") {
    forAll(Table(
      ("a", "b", "c"),
      (0.01, -1.5, 1.0)
      // (1.0, -1.0),
      // (0.05, 1.05),
      // (3.0, 3.0)
    )) { (a: Double, b: Double, c: Double) =>
      val nPoints = 1000

      val numIterations = 200
      val rho = 0.5
      val cee = 1.000

      // NOTE: Intercept should be small for generating equal 0s and 1s
      val testData = generateSVMInput(a, Array(b, c), nPoints, 42)

      val testRDD = sc.parallelize(testData, 8)
      testRDD.cache()

      val svm = new SVMWithADMM(numIterations = numIterations, rho = rho, cee = cee)

      val model = svm.run(testRDD)


      val validationData = generateSVMInput(a, Array(b, c), nPoints, 17)
      val validationRDD  = sc.parallelize(validationData, 2)

      // Test prediction on RDD.
      validatePrediction(model.predict(validationRDD.map(_.features)).collect(), validationData)

      val near = (v: Double) => v plusOrMinus(max(0.05, v * 0.05))
      model.intercept should be (near(a))
      model.weights(0) should be (near(b))


    }
  }
}

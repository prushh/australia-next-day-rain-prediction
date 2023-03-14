package it.unibo.andrp
package algorithm.tree

import it.unibo.andrp.model.DataPoint
import model.Feature
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


class DecisionTreeMR(maxDepth: Int = 3, featureSubsetStrategy: String = "all", minSplitSize: Int = 5, impurityMeasure: String = "gini") extends Serializable {

  private var root: Option[Node] = None


  private def getFeatureSubset(features: Seq[Int]): Seq[Int] = {
    featureSubsetStrategy match {
      case "all" => features
      case "sqrt" => Random.shuffle(features).take(Math.sqrt(features.length).ceil.toInt)
      case "log2" => Random.shuffle(features).take((Math.log(features.length) / Math.log(2.0)).ceil.toInt)
      case _ => throw new IllegalArgumentException(s"Invalid feature subset strategy: $featureSubsetStrategy")
    }
  }

  private def getImpurityMeasure(): ((Seq[Double], Seq[Double]) => Double) = {
    impurityMeasure match {
      case "gini" => calculateWeightedGiniIndex
      case "entropy" => entropyGainRatio
      case _ => throw new IllegalArgumentException(s"Invalid impurity measure: $featureSubsetStrategy")
    }
  }

  def filterFeatures(data: Seq[DataPoint], featureIndices: Seq[Int]): Seq[DataPoint] = {
    data.map(dp => dp.copy(features = featureIndices.map(i => dp.features(i)).toList))
  }

  def train(data: RDD[DataPoint], weights: Option[Seq[Double]]): Unit = {
    //feature selection initialization
    val features = data.collect().head.features.indices
    val featureSubset = getFeatureSubset(features)
    val data_filtered = filterFeatures(data.collect(), featureSubset)

    //weights intialization if there's no weight vector
    val w = weights match {
      case Some(x) => x
      case None => List.fill(data.collect().size)(1.0)
    }

    val impurityFunc = getImpurityMeasure()
    root = Some(buildDecisionTree(data, w, weightedImpurity, maxDepth, minSplitSize))
  }

  def predict(dataPoint: DataPoint): Double = {
    root match {
      case Some(node) => node.predict(dataPoint)
      case None => throw new Exception("Tree has not been trained yet")
    }
  }


  def buildDecisionTree(data: RDD[DataPoint],
                        weights: Seq[Double],
                        impurityFunc: (Seq[Double], Seq[Double]) => Double,
                        maxDepth: Int,
                        minSplitSize: Int
                       ): Node = {
    val labels = data.collect().map(_.label.toInt)
    if (maxDepth == 0 || labels.size < minSplitSize) {
      Leaf(getMajorityWeighted(labels, weights))
    } else {
      val (bestFeature, bestSplits, bestGain) = selectBestFeatureMapReduce(data, generateFeatures(data.collect().toList), weights, impurityFunc)
      if (bestGain == 0.0) {
        Leaf(getMajorityWeighted(labels, weights))
      } else {
        val spark = SparkContext.getOrCreate()
        val (leftData, leftWeights, rightData, rightWeights) = splitData(data.collect(), weights, bestFeature, bestSplits)
        val leftChild = buildDecisionTree(spark.parallelize(leftData), leftWeights, impurityFunc, maxDepth - 1, minSplitSize)
        val rightChild = buildDecisionTree(spark.parallelize(rightData), rightWeights, impurityFunc, maxDepth - 1, minSplitSize)
        InternalNode(bestFeature, bestSplits, leftChild, rightChild)
      }
    }
  }


  private def splitData(
                         data: Seq[DataPoint],
                         weights: Seq[Double],
                         feature: Int,
                         split: Double
                       ): (Seq[DataPoint], Seq[Double], Seq[DataPoint], Seq[Double]) = {
    val leftData = ArrayBuffer.empty[DataPoint]
    val leftLabels = ArrayBuffer.empty[Double]
    val leftWeights = ArrayBuffer.empty[Double]
    val rightData = ArrayBuffer.empty[DataPoint]
    val rightLabels = ArrayBuffer.empty[Double]
    val rightWeights = ArrayBuffer.empty[Double]

    for (((values, label), weight) <- data.map(_.features).zip(data.map(_.label)).zip(weights)) {
      if (values(feature) < split) {
        leftData += DataPoint(values, label)
        leftLabels += label
        leftWeights += weight
      } else {
        rightData += DataPoint(values, label)
        rightLabels += label
        rightWeights += weight
      }
    }

    (leftData.toList, leftWeights.toList, rightData.toList, rightWeights.toList)
  }

  // Calcola la miglior feature utilizzando MapReduce
  def selectBestFeatureMapReduce(data: RDD[DataPoint], features: Seq[Feature], weights: Seq[Double], impurityFunc: (Seq[Double], Seq[Double]) => Double
                                ):
  (Int, Double, Double) = {

    val featureValuesRDD = data.flatMap { dp =>
      features.map { feature =>
        (feature.id, dp.features(feature.id))
      }
    }
    val dataSeq = data.collect.toSeq
    val impurityReductionsRDD = featureValuesRDD.groupByKey().flatMap { case (featureId, values) =>

      values.toArray.sorted.distinct.sliding(2).map { case Array(v1, v2) =>{
        val threshold = (v1 + v2) / 2
        (featureId, threshold, calculateImpurityReduction(dataSeq, features(featureId), weights, threshold, impurityFunc))}
      case Array(v1) => {
        (featureId,v1,0.0)}
      }
    }
    val res =impurityReductionsRDD.collect().maxBy(_._3)
    res
  }

  def generateFeatures(data: List[DataPoint]): List[Feature] = {
    val numFeatures = data.head.features.length
    val featureValues = for {
      featureIndex <- 0 until numFeatures
    } yield Feature(featureIndex, data.map(_.features(featureIndex)).distinct.sorted)

    featureValues.toList
  }

  def calculateImpurityReduction(data: Seq[DataPoint], feature: Feature, weights: Seq[Double], value: Double, impurityFunc: (Seq[Double], Seq[Double]) => Double): Double = {
    val data_w = data zip weights
    val (left, right) = data_w.partition { dp =>
      dp._1.features(feature.id) <= value
    }

    val leftWeight = left.map(_._2).sum
    val rightWeight = right.map(_._2).sum


    val impurityReduction = impurityFunc(data.map(_.label), weights)- (leftWeight / weights.sum) * impurityFunc(left.map(_._1.label), left.map(_._2)) - (rightWeight / weights.sum) * impurityFunc(right.map(_._1.label), right.map(_._2))
    impurityReduction
  }
  def findBestWeightedSplit(
                             featureValues: Seq[Double],
                             labels: Seq[Double],
                             weights: Seq[Double],
                             impurityFunc: (Seq[Double], Seq[Double]) => Double): (Double, Double) = {

    val data = featureValues.zip(labels).zip(weights).sortBy(_._1._1)
    val totalWeight = weights.sum
    var leftWeight = 0.0
    var leftCount = 0.0
    var rightWeight = totalWeight
    var rightCount = labels.length.toDouble
    var bestSplit = Double.MinValue
    var bestGain = Double.MinValue

    for (((value, label), weight) <- data.dropRight(1)) {
      leftWeight += weight
      leftCount += 1
      rightWeight -= weight
      rightCount -= 1
      if (value < data.head._1._1 || value > data.last._1._1 || value == data.head._1._1) {
        // Skip values that are equal to the first or last value, or that are duplicates of the first value
        // (because they would result in an empty left or right subset)
      } else {
        val gain = impurityFunc(labels, weights) -
          (leftWeight / totalWeight) * impurityFunc(labels.take(leftCount.toInt), weights.take(leftCount.toInt)) -
          (rightWeight / totalWeight) * impurityFunc(labels.drop(leftCount.toInt), weights.drop(leftCount.toInt))
        if (gain > bestGain) {
          bestSplit = value
          bestGain = gain
        }
      }
    }
    (bestSplit, bestGain)
  }


  // Calcola la riduzione di impurezza per un attributo e un valore


  def findBestFeature(

                       data: Seq[DataPoint],
                       weights: Seq[Double],
                       impurityFunc: (Seq[Double], Seq[Double]) => Double
                     ): (Int, Double, Double) = {

    val numFeatures = data.head.features.size
    var bestFeature = -1
    var bestGain = -1000.0
    var bestSplits = -1000.0
    val labels = data.map(_.label)
    for (feature <- 0 until numFeatures) {
      val values = data.map(_.features(feature))

      val (splits, gain) = findBestWeightedSplit(values, labels, weights, impurityFunc)

      if (gain > bestGain) {
        bestFeature = feature
        bestGain = gain
        bestSplits = splits
      }
    }

    (bestFeature, bestGain, bestSplits)
  }





  def findBestWeightedSplit2(values: Seq[Double], labels: Seq[Int], weights: Seq[Double], impurityFunc: (Seq[Int], Seq[Double]) => Double): (Double, Double) = {
    val sortedData = values.zip(labels).zip(weights).sortBy(_._1._1)
    val weightsSum = weights.sum
    var leftWeightSum = 0.0
    var leftLabelSum = 0.0
    var bestGain = Double.NegativeInfinity
    var bestThreshold = Double.NaN

    // Ricerca binaria del threshold
    for (i <- 0 until values.size - 1) {
      val currentWeight = sortedData(i)._2
      leftWeightSum += currentWeight
      leftLabelSum += sortedData(i)._1._2 * currentWeight
      val rightWeightSum = weightsSum - leftWeightSum
      val rightLabelSum = sortedData.map(x => x._1._2 * x._2).sum - leftLabelSum
      val impurity = impurityFunc(Seq(leftLabelSum.toInt, rightLabelSum.toInt), Seq(leftWeightSum, rightWeightSum))
      if (impurity > bestGain) {
        bestGain = impurity
        bestThreshold = (sortedData(i)._1._1 + sortedData(i + 1)._1._1) / 2.0
      }
    }

    (bestThreshold, bestGain)
  }


  def accuracy(dataset: List[DataPoint]): Double = {
    val predictions = dataset.map(this.predict)
    val correct = predictions.zip(dataset).count { case (pred, dp) => pred == dp.label }
    correct.toDouble / dataset.length.toDouble
  }

  sealed trait Node {
    def predict(dataPoint: DataPoint): Int


  }

  case class Leaf(label: Int) extends Node {

    def predict(dataPoint: DataPoint): Int = label
  }

  case class InternalNode(feature: Int, threshold: Double, left: Node, right: Node) extends Node {
    def predict(dataPoint: DataPoint): Int =
      if (dataPoint.features(feature) <= threshold) left.predict(dataPoint) else right.predict(dataPoint)
  }


  def printTree(node: Node, indent: String = ""): Unit = {
    node match {
      case Leaf(label) => println(s"$indent Prediction: $label")
      case InternalNode(feature, threshold, left, right) =>
        println(s"$indent Feature $feature < $threshold")
        printTree(left, indent + "    ")
        println(s"$indent Feature $feature >= $threshold")
        printTree(right, indent + "    ")
    }
  }

  def printDecisionTree(): Unit = {
    println("Decision Tree")
    println("-------------")
    println(s"Maximum depth: ${this.maxDepth}")
    println("")

    printTree(this.root.get)
  }


  //gini impurity
  def weightedImpurity(labels: Seq[Double], weights: Seq[Double]): Double = {
    val weightedCounts = labels.zip(weights).groupBy(_._1).mapValues(_.map(_._2).sum)
    val totalWeight = weights.sum
    weightedCounts.values.map { count =>
      val weight = count / totalWeight
      weight * (1.0 - weight)
    }.sum
  }

  def entropyGainRatio(labels: Seq[Double], weights: Seq[Double]): Double = {
    val totalWeight = weights.sum
    val probabilities = labels.groupBy(identity).mapValues(_.size.toDouble / labels.size)
    val entropy = probabilities.values.map(p => -1 * p * Math.log(p)).sum * totalWeight
    if (entropy == 0) 0 else {
      val intrinsicValue = probabilities.values.map(p => -1 * p * Math.log(p)).sum * -1
      entropy / intrinsicValue
    }
  }

  def getMajorityWeighted(labels: Seq[Int], weights: Seq[Double]): Int = {
    val labelWeights = labels.zip(weights)
    val labelCounts = labelWeights.groupBy(_._1).mapValues(_.map(_._2).sum).toMap
    val (majorityLabel, _) = labelCounts.maxBy(_._2)
    majorityLabel
  }

  def calculateWeightedGiniIndex(label: Seq[Double],weights:Seq[Double]): Double = {

    val sc = SparkContext.getOrCreate()
    val dataRDD = sc.parallelize(label zip weights)

    // Fase di Map
    val classCounts = dataRDD//.map(dataPoint => (dataPoint._1, dataPoint._2))
      .reduceByKey(_ + _)
      .collectAsMap()

    val totalWeight = weights.sum

    // Fase di Reduce
    val giniImpurities = classCounts.mapValues(weightedCount => {
      val fraction = weightedCount / totalWeight
      fraction * (1 - fraction)
    }).values

    giniImpurities.sum
  }


}

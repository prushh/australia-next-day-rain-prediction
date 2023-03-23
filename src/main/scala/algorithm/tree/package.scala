package it.unibo.andrp
package algorithm

import model.DataPoint
import config.AlgorithmConfig
import algorithm.tree.Impurities.ImpurityFunc

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

package object tree {
//    def classify(sparkContext: SparkContext)(trainData: RDD[DataPoint])(weights: Option[Seq[Double]] = None): Unit = {
//        val trainDataArray = trainData.collect()
//
//        val size = trainDataArray.length
//        val features = trainDataArray.head.features.indices
//
//        val featureSubset = AlgorithmConfig.DecisionTree.SUBSET_STRATEGY match {
//            case SubsetStrategies.All => SubsetStrategies.all(features)
//            case SubsetStrategies.Sqrt => SubsetStrategies.sqrt(features)
//            case SubsetStrategies.Log2 => SubsetStrategies.log2(features)
//        }
//
//        val initWeights = _initWeights(weights, size)
//
//        val impurity: ImpurityFunc = AlgorithmConfig.DecisionTree.IMPURITY_MEASURE match {
//            case Impurities.Gini => Impurities.gini
//            case Impurities.Entropy => Impurities.entropy
//        }
//
//        val tree = algorithm.tree.DecisionTree
//        tree.setRoot(
//            Some(_build(trainData, initWeights, impurity))
//        )
//    }
//
//    private def _initWeights(weights: Option[Seq[Double]], length: Int): Seq[Double] = {
//        weights match {
//            case Some(x) => x
//            case None => Seq.fill(length)(1.0)
//        }
//    }
//
//    private def _build(trainData: RDD[DataPoint], weights: Seq[Double], impurity: ImpurityFunc): Node = {
//        val featureSelector = 10
//
//        //        val findFeatures = getSelectorFeatures()
//        //        val labels = data.collect().map(_.label.toInt)
//        //        if (maxDepth == 0 || labels.size < minSplitSize) {
//        //            Leaf(getMajorityWeighted(labels, weights))
//        //        } else {
//        //            val (bestFeature, bestSplits, bestGain) = selectBestFeatureMapReduce2(data, generateFeatures(data.collect().toList), weights, impurityFunc)
//        //            if (bestGain == 0.0) {
//        //                Leaf(getMajorityWeighted(labels, weights))
//        //            } else {
//        //                val spark = SparkContext.getOrCreate()
//        //                val (leftData, leftWeights, rightData, rightWeights) = splitData(data.collect(), weights, bestFeature, bestSplits)
//        //                val leftChild = buildDecisionTree(spark.parallelize(leftData), leftWeights, impurityFunc, maxDepth - 1, minSplitSize)
//        //                val rightChild = buildDecisionTree(spark.parallelize(rightData), rightWeights, impurityFunc, maxDepth - 1, minSplitSize)
//        //                InternalNode(bestFeature, bestSplits, leftChild, rightChild)
//        //            }
//        //        }
//    }
//
//    def _bestFeature(trainData: RDD[DataPoint], weights: Seq[Double], impurity: ImpurityFunc): (Int, Double, Double) = {
//        val features = _generate(trainData)
//        val featureValues = trainData.flatMap { dp =>
//            features.map { feature =>
//                (feature.id, dp.features(feature.id))
//            }
//        }
//
//        val trainDataArray = trainData.collect()
//        val impurityReductionsRDD = featureValues.groupByKey().flatMap {
//            case (featureId, values) =>
//                values.toArray.sorted.distinct.sliding(2).map {
//                    case Array(v1, v2) => {
//                        val threshold = (v1 + v2) / 2
//                        (featureId, threshold, calculateImpurityReduction(dataSeq, features(featureId), weights, threshold, impurity))
//                    }
//                    case Array(v1) => {
//                        (featureId, v1, 0.0)
//                    }
//                }
//        }
//        println(impurityReductionsRDD.collect().length)
//        val res = impurityReductionsRDD.collect().maxBy(_._3)
//        res
//    }
//
//    private def _generate(trainData: Seq[DataPoint]): Seq[Feature] = {
//        val numFeatures = traininData.head.features.length
//
//        val featureValues = (0 until numFeatures).map { idx =>
//            Feature(idx, trainData.map(_.features(idx)).distinct.sorted)
//        }
//
//        featureValues
//    }
//
//    def calculateImpurityReduction(data: Seq[DataPoint], feature: Feature, weights: Seq[Double], threshold: Double, impurityFunc: (Seq[Double], Seq[Double]) => Double): Double = {
//        val data_w = data zip weights
//        val (left, right) = data_w.partition { dp =>
//            dp._1.features(feature.id) <= threshold
//        }
//
//        val leftWeight = left.map(_._2).sum
//        val rightWeight = right.map(_._2).sum
//
//
//        val impurityReduction = impurityFunc(data.map(_.label), weights) - (leftWeight / weights.sum) * impurityFunc(left.map(_._1.label), left.map(_._2)) - (rightWeight / weights.sum) * impurityFunc(right.map(_._1.label), right.map(_._2))
//        impurityReduction
//    }
}

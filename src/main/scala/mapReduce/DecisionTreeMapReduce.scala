package it.unibo.andrp
package mapReduce

import Main.calculateAccuracy

import it.unibo.andrp.DataPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.SparkContext

class DecisionTreeMapReduce(sc: SparkContext, numCluster: Int = 5, featureSubsetStrategy: String = "all", minSplitSize: Int = 5, impurityMeasure: String = "gini") extends MapReduceTreeMethods {
  val fraction: Double = 1.0 / numCluster
  var trees = List.fill(numCluster)(new DecisionTreeGB(maxDepth = 3, featureSubsetStrategy, minSplitSize, impurityMeasure))

  override def train(train_data: Seq[DataPoint]): Unit = {
    val data_mapred = sc.parallelize(train_data)

    // Creazione degli alberi
    trees = trees.map { tree =>
      val sample = data_mapred.sample(true, fraction)
      tree.train(sample.collect().toList, None)
      tree
    }
  }

  // Funzione di predizione
  override def predict(point: DataPoint): Double = {
    val predictions = trees.map(tree => tree.predict(point))
    predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  override def accuracy(test_data: Seq[DataPoint]): Double = {
    //Make predictions on the testing data
    val predictions_map = test_data.map(predict)

    //Evaluate the accuracy of the predictions
    calculateAccuracy(predictions_map, test_data.map(_.label))
  }


}

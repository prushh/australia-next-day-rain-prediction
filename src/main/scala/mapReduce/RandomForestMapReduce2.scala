package it.unibo.andrp
package mapReduce

import org.apache.spark.SparkContext

class RandomForestMapReduce2(sc: SparkContext, numTrees: Int = 5, featureSubsetStrategy: String = "all", minSplitSize: Int = 5, impurityMeasure: String = "gini", maxDepth: Int = 5) extends MapReduceTreeMethods {

  private var trees: List[DecisionTreeGB] =
    List.fill(numTrees)(new DecisionTreeGB(maxDepth, featureSubsetStrategy = "sqrt"))

  override def train(train_data: Seq[DataPoint]): Unit = {
    val data_mapred = sc.parallelize(train_data)

    // Creazione degli alberi
    trees = trees.map { tree =>
      val sample = data_mapred.sample(true, 1.0)
      tree.train(sample.collect().toList,None)
      tree
    }
  }

  // Funzione di predizione
  override def predict(point: DataPoint): Double = {
    val predictions = trees.map(rf => rf.predict(point))
    predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  override def accuracy(test_data: Seq[DataPoint]): Double = {
    //Make predictions on the testing data
    val predictions_map = test_data.map(predict)

    //Evaluate the accuracy of the predictions
    calculateAccuracy(predictions_map, test_data.map(_.label))
  }

}

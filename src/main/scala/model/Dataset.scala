package it.unibo.andrp
package model

case class Dataset(dataPoints: Seq[DataPoint]) {
    def count(): Int = {
        dataPoints.length
    }
    
    def filterByLabel(label: Double): Dataset = {
        val filteredPoints = dataPoints.filter(_.label == label)
        Dataset(filteredPoints)
    }
}

//object Dataset {
//    def splitData(data: Seq[LabeledPoint], trainRatio: Double): (Dataset, Dataset) = {
//        val Array(trainData, testData) = data.randomSplit(Array(trainRatio, 1 - trainRatio))
//        val trainDataset = Dataset(trainData)
//        val testDataset = Dataset(testData)
//        (trainDataset, testDataset)
//    }
//}

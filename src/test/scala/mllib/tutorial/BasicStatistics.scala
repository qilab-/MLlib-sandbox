package mllib.tutorial

import mllib.helper.SparkHelper
import org.scalatest.{FunSpec, Matchers}

class BasicStatistics
  extends FunSpec
  with Matchers
  with SparkHelper {

  describe("Summary statistics") {
    it("Summary statistics") {
      import org.apache.spark.mllib.linalg.{Vector, Vectors}
      import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val v1: Vector = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0)
        val v2: Vector = Vectors.dense(2.0, 3.0, 4.0, 5.0, 6.0)
        val v3: Vector = Vectors.dense(3.0, 4.0, 5.0, 6.0, 7.0)
        val v4: Vector = Vectors.dense(4.0, 5.0, 6.0, 7.0, 8.0)
        val observations: RDD[Vector] = sc.parallelize(Seq(v1, v2, v3, v4))
        // Compute column summary statistics.
        val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)

        summary.mean should === (Vectors.dense(2.5, 3.5, 4.5, 5.5, 6.5))  // mean vector
      }
    }
  }

}

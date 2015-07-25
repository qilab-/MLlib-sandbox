package mllib.tutorial

import mllib.helper.SparkHelper
import org.scalatest.{FunSpec, Matchers}

class DataTypesSpec
  extends FunSpec
  with Matchers
  with SparkHelper {

  describe("Local vector") {
    it("vector") {
      import org.apache.spark.mllib.linalg.{Vector, Vectors}

      // Create a dense vector (1.0, 0.0, 3.0)
      val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
      // Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
      val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
      // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
      val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))

      dv should === (sv1)
      dv should === (sv2)
    }
  }

  describe("Labeled point") {
    it("labeled point") {
      // For binary classification, a label should be either 0 (negative) or 1 (positive).
      // For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....
      import org.apache.spark.mllib.linalg.Vectors
      import org.apache.spark.mllib.regression.LabeledPoint

      // Create a labeled point with a positive label and a dense feature vector.
      val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
      // Create a labeled point with a negative label and a sparse feature vector.
      val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))

      pos.label should equal (1)  // positive
      neg.label should equal (0)  // negative
      pos.features should === (Vectors.dense(1.0, 0.0, 3.0))  // feature vector
      neg.features should === (Vectors.dense(1.0, 0.0, 3.0))  // feature vector
    }

    it("reading training examples stored in LIBSVM format") {
      import org.apache.spark.SparkContext
      import org.apache.spark.mllib.linalg.Vectors
      import org.apache.spark.mllib.regression.LabeledPoint
      import org.apache.spark.mllib.util.MLUtils
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "src/test/resources//iris_6.scale")
        val expected: Seq[LabeledPoint] = Seq(
          LabeledPoint(1.0, Vectors.dense(-0.555556, 0.25, -0.864407, -0.916667)),
          LabeledPoint(1.0, Vectors.dense(-0.611111, 0.0833333, -0.864407, -0.916667)),
          LabeledPoint(2.0, Vectors.dense(0.5, 0.0, 0.254237, 0.0833333)),
          LabeledPoint(2.0, Vectors.dense(-0.222222, -0.333333, 0.0508474, -4.03573e-08)),
          LabeledPoint(3.0, Vectors.dense(0.111111, 0.0833333, 0.694915, 1)),
          LabeledPoint(3.0, Vectors.dense(-0.111111, -0.166667, 0.38983, 0.416667))
        )

        examples.collect should have size expected.size
        examples.collect should equal (expected)
      }
    }
  }

  describe("Local matrix") {
    it("local matrix") {
      import org.apache.spark.mllib.linalg.{Matrix, Matrices}

      // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
      val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))
      dm.numCols should equal (2)
      dm.numRows should equal (3)
      dm(0, 0) should equal (1.0)
      dm(0, 1) should equal (2.0)
      dm(1, 0) should equal (3.0)
      dm(1, 1) should equal (4.0)
      dm(2, 0) should equal (5.0)
      dm(2, 1) should equal (6.0)
    }
  }

  describe("Distributed matrix") {
    it("RowMatrix") {
      // A RowMatrix is a row-oriented distributed matrix without meaningful row indices.
      import org.apache.spark.mllib.linalg.{Vector, Vectors}
      import org.apache.spark.mllib.linalg.distributed.RowMatrix
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val rv: Vector = Vectors.dense(-0.555556, 0.25, -0.864407, -0.916667)
        val rows: RDD[Vector] = sc.parallelize(Seq(rv))  // an RDD of local vectors
        // Create a RowMtrix from an RDD[Vector].
        val mat: RowMatrix = new RowMatrix(rows)

        // Get its size.
        mat.numRows should equal (1)
        mat.numCols should equal (rv.size)
      }
    }

    it("IndexedRowMatix") {
      // An IndexedRowMatrix is similar to a RowMatrix but with meaningful row indices.
      import org.apache.spark.mllib.linalg.{Vector, Vectors}
      import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val irs: Seq[IndexedRow] = Seq(
          new IndexedRow(0L, Vectors.dense(-0.555556, 0.25, -0.864407, -0.916667)),
          new IndexedRow(5L, Vectors.dense(-0.111111, -0.166667, 0.38983, 0.416667))
        )
        val rows: RDD[IndexedRow] = sc.parallelize(irs)  // an RDD of indexed rows
        // Create an IndexedRowMatrix from an RDD[IndexedRow]
        val mat: IndexedRowMatrix = new IndexedRowMatrix(rows)

        mat.numRows should equal (6)  // max index (5) + 1
        mat.numCols should equal (4)

        // Drop its row indices.
        val rowMat: RowMatrix = mat.toRowMatrix
        rowMat.numRows should equal (2)
        rowMat.numCols should equal (4)
      }
    }

    it("CoordinateMatrix") {
      // A CoordinateMatrix is a distributed matrix backed by an RDD of its entries.
      // A CoordinateMatrix should be used only when both dimensions of the matrix are huge and the matrix is very sparse.
      import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val me1: MatrixEntry = new MatrixEntry(3L, 5L, 2.0)
        val me2: MatrixEntry = new MatrixEntry(6L, 8L, 4.0)
        val entries: RDD[MatrixEntry] = sc.parallelize(Seq(me1, me2))  // an RDD of matrix entries
        // Create a CoordinateMatrix from an RDD[MatrixEntry].
        val mat: CoordinateMatrix = new CoordinateMatrix(entries)

        // Get its size.
        mat.numRows should equal (7)  // max row index (6) + 1
        mat.numCols should equal (9)  // max col index (8) + 1

        // Convert it to an IndexRowMatrix whose rows are sparse vectors.
        val indexedRowMatrix = mat.toIndexedRowMatrix
        indexedRowMatrix.numRows should equal (7)
        indexedRowMatrix.numCols should equal (9)

        val rowMat = indexedRowMatrix.toRowMatrix
        rowMat.numRows should equal (2)
        rowMat.numCols should equal (9)
      }
    }

    it("BlockMatrix") {
      // A BlockMatrix is a distributed matrix backed by an RDD of MatrixBlocks,
      // where a MatrixBlock is a tuple of ((Int, Int), Matrix),
      // where the (Int, Int) is the index of the block,
      // and Matrix is the sub-matrix at the given index with size rowsPerBlock x colsPerBlock.
      import org.apache.spark.mllib.linalg.{Matrix, Matrices, Vectors}
      import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, MatrixEntry}
      import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
      import org.apache.spark.rdd.RDD

      withSparkContext { sc =>
        val me1: MatrixEntry = new MatrixEntry(3L, 5L, 2.0)
        val me2: MatrixEntry = new MatrixEntry(6L, 8L, 4.0)
        val entries: RDD[MatrixEntry] = sc.parallelize(Seq(me1, me2))  // an RDD of (i, j, v) matrix entries
        // Create a CoordinateMatrix from an RDD[MatrixEntry].
        val coordMat: CoordinateMatrix = new CoordinateMatrix(entries)
        // Transform the CoordinateMatrix to a BlockMatrix
        val matA: BlockMatrix = coordMat.toBlockMatrix.cache

        // Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
        // Nothing happens if it is valid.
        matA.validate()

        // Calculate A^T A.
        val ate = matA.transpose.multiply(matA)

        // Create a MatrixBlock that is a tuple of ((Int, Int), Matrix)
        val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

        // pattern 1
        // 0.0  0.0  1.0  2.0
        // 0.0  0.0  3.0  4.0
        // 0.0  0.0  5.0  6.0
        // 1.0  2.0  0.0  0.0
        // 3.0  4.0  0.0  0.0
        // 5.0  6.0  0.0  0.0
        val matBlock11: ((Int, Int), Matrix) = ((0, 1), dm)
        val matBlock12: ((Int, Int), Matrix) = ((1, 0), dm)
        val blocks1 = sc.parallelize(Seq(matBlock11, matBlock12))
        val matB1: BlockMatrix = new BlockMatrix(blocks1, 3, 2, 6, 4)
        matB1.validate()
        val irsEx1: Seq[IndexedRow] = Seq(
          new IndexedRow(0L, Vectors.dense(0, 0, 1, 2)),
          new IndexedRow(1L, Vectors.dense(0, 0, 3, 4)),
          new IndexedRow(2L, Vectors.dense(0, 0, 5, 6)),
          new IndexedRow(3L, Vectors.dense(1, 2, 0, 0)),
          new IndexedRow(4L, Vectors.dense(3, 4, 0, 0)),
          new IndexedRow(5L, Vectors.dense(5, 6, 0, 0))
        )
        val rowsEx1: RDD[IndexedRow] = sc.parallelize(irsEx1)  // an RDD of indexed rows
        val matEx1: Matrix = new IndexedRowMatrix(rowsEx1).toBlockMatrix.toLocalMatrix
        matB1.toLocalMatrix should equal (matEx1)

        // pattern 2
        // 0.0  0.0  0.0  0.0
        // 0.0  0.0  0.0  0.0
        // 0.0  0.0  0.0  0.0
        // 0.0  0.0  1.0  2.0
        // 0.0  0.0  3.0  4.0
        // 0.0  0.0  5.0  6.0
        val matBlock21: ((Int, Int), Matrix) = ((1, 1), dm)
        val blocks2 = sc.parallelize(Seq(matBlock21))
        val matB2: BlockMatrix = new BlockMatrix(blocks2, 3, 2, 6, 4)
        matB2.validate()
        val irsEx2: Seq[IndexedRow] = Seq(
          new IndexedRow(3L, Vectors.dense(0, 0, 1, 2)),
          new IndexedRow(4L, Vectors.dense(0, 0, 3, 4)),
          new IndexedRow(5L, Vectors.dense(0, 0, 5, 6))
        )
        val rowsEx2: RDD[IndexedRow] = sc.parallelize(irsEx2)  // an RDD of indexed rows
        val matEx2: Matrix = new IndexedRowMatrix(rowsEx2).toBlockMatrix.toLocalMatrix
        matB2.toLocalMatrix should equal (matEx2)
      }
    }
  }
}

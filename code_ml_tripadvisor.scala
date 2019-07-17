import org.apache.spark.sql.Row
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, StringIndexer, IndexToString}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RegexTokenizer

//import dataset in cache
var df=spark.read.format("csv").option("header","true").option("delimiter","|").load("/home/master/Desktop/inputprovapipe2_1207_5.csv")
var df_score=spark.read.format("csv").option("header","true").option("delimiter","|").load("/home/master/Desktop/dataset_score_1207.csv")
df.registerTempTable("mal")
var df_train=spark.sql("select * from mal where tipo_lista='TRAINING'")
var df_test=spark.sql("select * from mal where tipo_lista='TEST'")
df_test.cache()
df_train.cache()
df_score.cache()
df.show()
df_test.count()
df_train.count()
// indexer
val indexer = new StringIndexer().setInputCol("labelpers").setOutputCol("label").fit(df_train)
//tokenizer
val tokenizer = new Tokenizer().setInputCol("commento").setOutputCol("words")
//test
val tokenized = tokenizer.transform(df_train)
val tokens = tokenized.select(explode(col("words")).as("word")).groupBy("word").count().orderBy(desc("count"))
tokens.show(false)
// uso stopwordremover in italiano
val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("clean_words").setStopWords(StopWordsRemover.loadDefaultStopWords("italian"))
val cleaned = stopWordsRemover.transform(tokenized)
// check parole uniche  rimanenti
val tokens = cleaned.select(explode(col("clean_words")).as("word")).groupBy("word").count().orderBy(desc("count"))
tokens.show(false)
// check numero parole totali rimanenti
val tokens = cleaned.select(explode(col("clean_words")).as("word")).count()
// miglioro a tokenizzazione --> messo il pattern regex per parole italiane
val tokenizer = new RegexTokenizer().setInputCol("commento").setOutputCol("words").setPattern("\\p{P}|\\s")
val tokenized = tokenizer.transform(df_train)
val cleaned = stopWordsRemover.transform(tokenized)
val tokens = cleaned.select(explode(col("clean_words")).as("word")).groupBy("word").count().orderBy(desc("count"))
tokens.show(false)
// check numero parole totali rimanenti
val tokens = cleaned.select(explode(col("clean_words")).as("word")).count()
// hash e training modello
val dim = math.pow(2, 14).toInt
val hashingTF = new HashingTF().setNumFeatures(dim).setInputCol("clean_words").setOutputCol("rawFeatures")
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val nb  = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").setSmoothing(1.0).setModelType("multinomial")
val converter = new IndexToString().setInputCol("prediction").setOutputCol("predictedCategory").setLabels(indexer.labels)
val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, stopWordsRemover, hashingTF, idf, nb, converter))
val model = pipeline.fit(df_train)
// applicazione modello su dataset validazione e score
val validation = model.transform(df_test)
val score = model.transform(df_score)
val predictions = validation.select("labelpers", "predictedCategory")
predictions.shoW
val metrics = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
metrics.getMetricName
val metrics = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("label").setPredictionCol("prediction")
metrics.evaluate(validation)
//accuracy
val metrics = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision").setLabelCol("label").setPredictionCol("prediction")
metrics.evaluate(validation)
//precision
val metrics = new MulticlassClassificationEvaluator().setMetricName("weightedRecall").setLabelCol("label").setPredictionCol("prediction")
metrics.evaluate(validation)
//recall
// scarico i df finali
val predictions = validation.select("utente","commento","hotel","datains","tipo_lista","labelpers", "predictedCategory")
val score_out = score.select("utente","commento","hotel","datains","tipo_lista","labelpers", "predictedCategory")
val training_out = df_train.select("utente","commento","hotel","datains","tipo_lista","labelpers")
val training_out2 = training_out.withColumn("predictedCategory", training_out("labelpers")) 
val output1=predictions.union(score_out)
val output2=output1.union(training_out2)
output2.coalesce(1).write.option("header","true").option("delimiter","|").option("escape","\\").option("quoteAll","true").format("com.databricks.spark.csv").csv("/home/master/Desktop/output_12_07_v2.csv")
// estrazione parole per word_cloud
val words_input = cleaned.select("utente", "hotel","clean_words").withColumn("clean_words", explode(col("clean_words")))
val words_score = score.select("utente", "hotel","clean_words").withColumn("clean_words", explode(col("clean_words")))
val word_cloud = words_input.union(words_score)
word_cloud.coalesce(1).write.option("header","true").option("delimiter","|").option("escape","\\").option("quoteAll","true").format("com.databricks.spark.csv").csv("/home/master/Desktop/word_cloud.csv")

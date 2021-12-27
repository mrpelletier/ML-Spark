#Required Packages
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, LongType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import sum as _sum
from pyspark.sql.functions import lit
import sys
import codecs


# Create a SparkSession (the config bit is only for Windows!)
spark = SparkSession.builder.config(
    "spark.sql.warehouse.dir", 
    "file:///C:/temp").appName("PopularMovies").getOrCreate()


def loadMovieNames():
    movieNames = {}
    # CHANGE THIS TO THE PATH TO YOUR u.ITEM FILE:
    with codecs.open("u.ITEM", "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

""" ************** Define Schemas ************** """

#Build schema for u.Data (Note: "timestamp" not converted from unix)
schemaData = StructType([StructField("userID", IntegerType(), True),
                     StructField("movieID", IntegerType(), True),
                     StructField("rating", IntegerType(), True),
                     StructField("timestamp", LongType(), True)])

#Build schema for u.Item (Note: boolean values for genres listed as int)
schemaItem = StructType([StructField("movieID", IntegerType(), True), 
                     StructField("movie_title", StringType(), True), 
                     StructField("release_date", DateType(),True), 
                     StructField("video_release_date", DateType(),True), 
                     StructField("IMDb_URL", StringType(), True), 
                     StructField("unknown", IntegerType(), True), 
                     StructField("Action", IntegerType(), True), 
                     StructField("Adventure", IntegerType(), True), 
                     StructField("Animation", IntegerType(), True), 
                     StructField("Children's", IntegerType(), True), 
                     StructField("Comedy", IntegerType(), True), 
                     StructField("Crime", IntegerType(), True), 
                     StructField("Documentary", IntegerType(), True), 
                     StructField("Drama", IntegerType(), True), 
                     StructField("Fantasy", IntegerType(), True), 
                     StructField("Film-Noir", IntegerType(), True), 
                     StructField("Horror", IntegerType(), True), 
                     StructField("Musical", IntegerType(), True), 
                     StructField("Mystery", IntegerType(), True),
                     StructField("Romance", IntegerType(), True), 
                     StructField("Sci-Fi", IntegerType(), True), 
                     StructField("Thriller", IntegerType(), True), 
                     StructField("War", IntegerType(), True), 
                     StructField("Western", IntegerType(), True)])

#Build schema for u.user 
schemaUser = StructType([StructField("userID", IntegerType(), True), 
                     StructField("age", IntegerType(), True), 
                     StructField("gender", StringType(),True), 
                     StructField("occupation", StringType(),True), 
                     StructField("zip code", LongType(), True)])

""" ************** load Movie Names ************** """

names = loadMovieNames()
userAge = int(sys.argv[1])
""" ************** Create Data Frames and Import Data ************** """

#Create Data Frame for u.Data
dfDATA = spark.read.options(delimiter="\t").csv(
    "u.data",header=False,schema=schemaData)

#Create Data Frame for u.Item
dfITEM = spark.read.options(delimiter="|").csv(
    "u.ITEM",header=False,dateFormat="dd-MMM-yyyy",schema=schemaItem)

#Create Data Frame for u.User
dfUSER = spark.read.options(delimiter="|").csv(
    "u.user",header=False,schema=schemaUser)

#Show imported data
#dfDATA.show(5)
#dfITEM.show(5)
#dfUSER.show(5)


#Select required columns
dfDATA_1 = dfDATA.select("userID","movieID", "rating")
dfITEM_1 = dfITEM.drop("release_date","video_release_date","IMDb_URL")
dfUSER_1 = dfUSER.drop("zip code")


dfCombined = dfDATA_1.join(dfITEM_1,"movieID").join(dfUSER_1,"userID").withColumn("Count", lit(1))

dfCombined_1 = dfCombined.groupBy("age","movieID","movie_title").agg(_sum("rating").alias("total_rating"),_sum("Count").alias("Count"))

dfCombined_2 = dfCombined_1.withColumn("avg_Rating",dfCombined_1.total_rating/dfCombined_1.Count)

High_rated_Age_35 = dfCombined_2.filter(dfCombined_2.age == userAge).sort("avg_Rating", ascending=False).drop("total_rating")
print("*************** Movies with Highest Ratings for Age 35 ***************")
High_rated_Age_35.show(10)

Most_Pop_Age_35 = dfCombined_2.filter(dfCombined_2.age == userAge).sort("Count", ascending=False).drop("total_rating")
print("*************** Most Watched Movies for Age 35 ***************")
Most_Pop_Age_35.show(10)

""" ************** Set-up and run regression ************** """
(training, test) = dfCombined_2.randomSplit([0.8, 0.2])



""" Build Model """

als = ALS(maxIter=10, regParam=0.1, rank=7,userCol="age", itemCol="movieID", ratingCol="avg_Rating",
          coldStartStrategy="drop",
          nonnegative=True)

model=als.fit(training)

predictions=model.transform(test)

evaluator=RegressionEvaluator(metricName="rmse",labelCol="avg_Rating",
                              predictionCol="prediction")

rmse=evaluator.evaluate(predictions)

print("*************** RMSE ***************")
print("Root-mean-square error =" +str(rmse))
print("                                    ")

# Manually construct a dataframe of the user age we want recs for
userSchema = StructType([StructField("age", IntegerType(), True)])
users = spark.createDataFrame([[userAge,]], userSchema)

recommendations = model.recommendForUserSubset(users, 10).collect()

print("*************** Top 10 recommendations for user Age " + str(userAge) + " ***************")

for userRecs in recommendations:
    myRecs = userRecs[1]  #userRecs is (userID, [Row(movieId, rating), Row(movieID, rating)...])
    for rec in myRecs: #my Recs is just the column of recs for the user
        movie = rec[0] #For each rec in the list, extract the movie ID and rating
        rating = rec[1]
        movieName = names[movie]
        print(movieName + str(rating))


#Stop the session
spark.stop()

#Execution statment to use in IPython console
# !spark-submit Recommendations_for_every_Age.py 35

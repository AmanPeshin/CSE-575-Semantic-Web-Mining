//C:\Users\Aman-ASU\My Graduate\My Courses\Semantic Web Mining\Project\Graph-ML-Fraud-Detection\relate-data\dbmss\dbms-2731ca35-5871-4187-a635-fe5794dac250\import
//5.13.0 DB version || APOC version 5.13.0 || GDS version 2.5.5

CREATE CONSTRAINT FOR (c:Customer) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT FOR (m:Merchant) REQUIRE m.id IS UNIQUE

LOAD CSV WITH HEADERS FROM
"file:///bs140513_032310.csv" AS line
WITH line,
SPLIT(line.customer, "'") AS customerID,
SPLIT(line.merchant, "'") AS merchantID,
SPLIT(line.age, "'") AS customerAge,
SPLIT(line.gender, "'") AS customerGender,
SPLIT(line.category, "'") AS transCategory
MERGE (customer:Customer {id: customerID[1], age: customerAge[1], gender: customerGender[1]})
MERGE (merchant:Merchant {id: merchantID[1]})
CREATE (transaction:Transaction {amount: line.amount, fraud: line.fraud, category: transCategory[1], step: line.step})-[:WITH]->(merchant)
CREATE (customer)-[:PERFORMS]->(transaction);

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1
MERGE (p1:Placeholder {id: m1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MERGE (p2:Placeholder {id:c1.id})

// Create Placeholder nodes
MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt, 
     sum(toFloat(t1.amount)) as totalAmount, 
     collect(t1.fraud) as fraudList, 
     collect(t1.category) as categoryList
MERGE (p1:Placeholder {id: m1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt, 
     sum(toFloat(t1.amount)) as totalAmount, 
     collect(t1.fraud) as fraudList, 
     collect(t1.category) as categoryList
MERGE (p2:Placeholder {id:c1.id})

// Create PAYS relationships with additional features
MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt, 
     sum(toFloat(t1.amount)) as totalAmount, 
     collect(t1.fraud) as fraudList, 
     collect(t1.category) as categoryList
MATCH (p1:Placeholder {id:m1.id})
WITH c1, m1, p1, cnt, totalAmount, fraudList, categoryList
MATCH (p2:Placeholder {id: c1.id})
WITH c1, m1, p1, p2, cnt, totalAmount, fraudList, categoryList
CREATE (p2)-[:PAYS {
  cnt: cnt, 
  totalAmount: totalAmount, 
  fraudList: fraudList, 
  categoryList: categoryList
}]->(p1)

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt, 
     sum(toFloat(t1.amount)) as totalAmount, 
     collect(t1.fraud) as fraudList, 
     collect(t1.category) as categoryList
MATCH (p1:Placeholder {id:c1.id})
WITH c1, m1, p1, cnt, totalAmount, fraudList, categoryList
MATCH (p2:Placeholder {id: m1.id})
WITH c1, m1, p1, p2, cnt, totalAmount, fraudList, categoryList
CREATE (p1)-[:PAYS {
  cnt: cnt, 
  totalAmount: totalAmount, 
  fraudList: fraudList, 
  categoryList: categoryList
}]->(p2)

// Create a graph projection for your data
CALL gds.graph.project('banksim',{Placeholder: {label: 'Placeholder'}}, {PAYS: {type: 'PAYS', orientation: 'NATURAL'}});

// Computing local clustering coefficient
MATCH (n:Placeholder)-[:PAYS]-(m:Placeholder)
WHERE EXISTS((n)-[:PAYS]-(m))
WITH n, COUNT(*) AS degree
MATCH (n)-[:PAYS]-(m)-[:PAYS]-(p)
WHERE id(m) < id(p)
WITH n, degree, (2.0 * COUNT(DISTINCT p)) / (degree * (degree - 1)) AS localClusteringCoefficient
SET n.localClusteringCoefficient = localClusteringCoefficient;

MATCH (n:Placeholder)-[:PAYS]-(m:Placeholder)
WHERE EXISTS((n)-[:PAYS]-(m))
WITH m, COUNT(*) AS degree
MATCH (n)-[:PAYS]-(m)-[:PAYS]-(p)
WHERE id(m) < id(p)
WITH m, degree, (2.0 * COUNT(DISTINCT p)) / (degree * (degree - 1)) AS localClusteringCoefficient
SET m.localClusteringCoefficient = localClusteringCoefficient;

// View local clustering results
MATCH (p:Placeholder)
RETURN p.id AS id, p.localClusteringCoefficient AS localClusteringCoefficient
ORDER BY localClusteringCoefficient ASC;

// Compute PageRank and store the results in the 'pagerank' property
CALL gds.pageRank.write('banksim', {
  nodeLabels: ['Placeholder'],
  relationshipTypes: ['PAYS'],
  writeProperty: 'pagerank',
  maxIterations: 10000,
  dampingFactor: 0.85,
  tolerance: 0.0000001,
  scaler: "MEAN"
});

// View the PageRank results
MATCH (p:Placeholder)
RETURN p.id AS id, p.pagerank AS pagerank
ORDER BY pagerank DESC;

CALL gds.labelPropagation.write('banksim', {
  nodeLabels: ['Placeholder'],
  relationshipTypes: ['PAYS'],
  writeProperty: 'community',
  nodeWeightProperty: ''
});

// View the community results
MATCH (p:Placeholder)
RETURN p.id AS id, p.community AS community
ORDER BY community DESC;

// Computing the degree of each node
MATCH (p:Placeholder)
SET p.degree = apoc.node.degree(p, 'PAYS')

MATCH (p:Placeholder)
RETURN p.id AS id, p.degree AS degree
ORDER BY degree ASC;


// Compute node similarity and store the results in the 'similarity' property
CALL gds.nodeSimilarity.stream('banksim')
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS n1, gds.util.asNode(node2) AS n2, similarity
MERGE (n1)-[:SIMILAR { similarity: similarity }]->(n2);

// View the node similarity results
MATCH (n1)-[r:SIMILAR]-(n2)
RETURN n1.id AS node1, n2.id AS node2, r.similarity AS similarity
ORDER BY similarity DESC;

// Run Node2Vec algorithm
CALL gds.node2vec.write('banksim', {
  nodeLabels: ['Placeholder'],
  relationshipTypes: ['PAYS'],
  writeProperty: 'node2vec_embedding', // Change to your desired property name
  walkLength: 80,
  walksPerNode: 10,
  inOutFactor: 1.0,
  returnFactor: 1.0,
  relationshipWeightProperty: null,
  windowSize: 10,
  negativeSamplingRate: 5,
  positiveSamplingFactor: 0.001,
  negativeSamplingExponent: 0.75,
  embeddingDimension: 128,
  embeddingInitializer: 'NORMALIZED',
  iterations: 1,
  initialLearningRate: 0.01,
  minLearningRate: 0.0001,
  randomSeed: 'random',
  walkBufferSize: 1000
})
YIELD
  preProcessingMillis,
  computeMillis,
  writeMillis,
  nodeCount,
  nodePropertiesWritten,
  lossPerIteration,
  configuration;
  
//How to view results of Node2Vec, like graphSAGE but works with string data
MATCH (p:Placeholder)
RETURN p.id AS id, p.node2vec_embedding AS node2vec_embedding
ORDER BY node2vec_embedding ASC;
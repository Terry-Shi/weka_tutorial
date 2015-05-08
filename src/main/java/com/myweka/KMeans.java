package com.myweka;

import java.io.File;

import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.EuclideanDistance;


public class KMeans {
    
    public static void main(String[] args) {
    	try {
	    	ArffLoader loader = new ArffLoader();
	    	loader.setFile(new File("data/kmeans_no_normalize.arff"));
	    	
	    	Instances data =  loader.getDataSet();
	    	data.deleteStringAttributes();
	    	
	    	SimpleKMeans simpleKMeans = new SimpleKMeans();
	    	// 设置聚类要得到的类别数量
	    	simpleKMeans.setNumClusters(3);
	    	
	    	// 函数设置聚类算法内部的距离计算方式
	    	DistanceFunction m_DistanceFunction = new EuclideanDistance();
	    	simpleKMeans.setDistanceFunction(m_DistanceFunction);//default: weka.core.EuclideanDistance
	    	simpleKMeans.setMaxIterations(10);
	    	
	    	// 使用聚类算法对样本进行聚类
	    	simpleKMeans.buildClusterer(data);
	    	
	    	Instances tempIns = simpleKMeans.getClusterCentroids();
	        System.out.println("聚类中心:");
	        System.out.println("CentroIds: " + tempIns);
	        
	        System.out.println("------ Level ------");
	        int totalCount = data.numInstances();
	        for (int i=0; i<totalCount; i++) {
	        	Instance currIns = data.instance(i);
	        	System.out.println(simpleKMeans.clusterInstance(currIns)); //currIns + " <-> " + 
	        }
	        
//	        Instance ins = new Instance(data.firstInstance());
//	    	int i = simpleKMeans.clusterInstance(ins);
//	    	System.out.println("China Football is level " + i);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    
}

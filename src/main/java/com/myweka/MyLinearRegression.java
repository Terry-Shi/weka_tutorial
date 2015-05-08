package com.myweka;

import java.io.File;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * 
 * @author shijie
 * ref: sample code http://supercharles888.blog.51cto.com/609344/1358756
 * ref: GUI http://www.cnblogs.com/bourneli/archive/2012/10/15/2725019.html
 */
public class MyLinearRegression {
    public static void main(String[] args) throws Exception {
        String dataFile = "data/chepai.arff";
        //演示如何获取ARFF文件中样本数据定义格式信息
        String arffFileDef = parseArffFile(dataFile);
        System.out.println(arffFileDef);
        System.out.println();
                                 
        //演示如何从ARFF文件中挖掘数据联系（公式）以及给出数学概率的评定
        String evalResult = doLinearRegression(dataFile);
        System.out.println(evalResult);
    }
    
    /**
     * 分析ARFF文件，获取其文件中的格式定义信息
     * @param filePath  传入的ARFF文件的文件路径，这里暂时不支持http和ftp，只支持本地文件
     * @return          封装字符串的文件内容返回对象
     * @throws Exception
     */
    public static String parseArffFile(String filePath) throws Exception {
        // 创建一个arff文件载入器
        ArffLoader loader = new ArffLoader();
                                                                                   
        //载入文件内容，获取其数据集合
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
                                                                                   
        //封装字符串的文件内容返回对象
        StringBuilder sb = new StringBuilder();
        sb.append("被读取的训练文件路径为：" + filePath + "\n\n");
        sb.append("训练文件内容定义为:" + new Instances(data, 0));
        return sb.toString();
    }
    
    /**
     * 对ARFF文件中的数据集合做线性回归，从而挖掘出其中的公式
     * @param filePath    传入的ARFF文件的文件路径，这里暂时不支持http和ftp，只支持本地文件
     * @return            线性回归运算得到的公式，以及运算结果的评估
     * @throws Exception
     */
    public static String doLinearRegression(String filePath) throws Exception {
        // 读训练数据
        DataSource train_data = new DataSource(filePath);
        // 获取训练数据集
        Instances insTrain = train_data.getDataSet();
        // 设置训练集中，target的index
        insTrain.setClassIndex(insTrain.numAttributes() - 1);
        // 定义分类器的类型 , 我们采用线性回归
        LinearRegression lr = new LinearRegression();
        // 训练分类器
        lr.buildClassifier(insTrain);
        
        // 评估线性回归的结果
        Evaluation eval = new Evaluation(insTrain);
        eval.evaluateModel(lr, insTrain);// 评估结果
        // 构造结果对象
        StringBuilder sb = new StringBuilder();
        sb.append("机器学习后产生的线性回归公式:\n" + lr.toString() + "\n\n");
        sb.append("评估此结果:" + eval.toSummaryString() + "\n");
        return sb.toString();
    }
    
}

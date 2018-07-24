package com.java.inputmethod.latin;

import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/**
 * Created by croath on 26/05/2017.
 */
public class RNNModelTest {
    private static final int RESULT_LENGTH = 20;
    static RNNModel mModel;

    @BeforeClass
    public static void setUpBefore() throws Exception {
        System.load(System.getProperty("user.dir") +"/res/librnn_dict.so");
        mModel = new RNNModel();
        mModel.init();
    }

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
        return map.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                ));
    }

//    @Test
//    public void predictSmall() throws Exception {
//        predict("res/single-test.txt");
//    }

    @Test
    public void predictMedium() throws Exception {
        predict("res/10k.test");
    }

    public void predict(String fileName) throws Exception {
        BufferedReader mBuf = new BufferedReader(new FileReader(fileName));

        int in10 = 0;
        int in5 = 0;
        int in3 = 0;
        int in1 = 0;
        int allCount = 0;

        mModel.predict();
        for(String line; (line = mBuf.readLine()) != null; ) {
            String sentence = line.trim()
                    .replaceAll("[,;&:\\-\\$]+", "<pun>")
                    .replaceAll("[0-9]+", "<num>")
                    .replaceAll("[.?!]+", "<eos>");
            String[] sourceWords = sentence.trim().split(" ");
//            mModel.predict();
            mModel.predict("<eos>", null, null);
            for (int i = 1; i < sourceWords.length - 1; i++) {

                String expectedResult = sourceWords[i];
                String inputWord = sourceWords[i-1];
                String[] results = new String[RESULT_LENGTH];
                int[] scores = new int[RESULT_LENGTH];
                mModel.predict(inputWord, results, scores);

                if (expectedResult.equals("<pun>") || expectedResult.equals("<num>") || expectedResult.equals("<eos>")) {
                    continue;
                }

                allCount++;

                Map<String, Integer> resDict = new HashMap<String, Integer>();

                for (int j = 0; j < RESULT_LENGTH; j++) {
                    String word = results[j];
                    if (word == null) {
                        continue;
                    }
                    int score = scores[j];
                    if (resDict.containsKey(word)) {
                        resDict.put(word, resDict.get(word) + score);
                    } else {
                        resDict.put(word, score);
                    }
                }

                resDict = sortByValue(resDict);
                String[] resStrs = resDict.keySet().toArray(new String[resDict.keySet().size()]);

                for (int j = 0; j < Math.min(resStrs.length, 10); j++) {
                    String str = resStrs[j];
                    if (str == null) {
                        continue;
                    }

                    if (str.equals(expectedResult)) {
                        in10 ++;
                        if (j < 5) in5 ++;
                        if (j < 3) in3 ++;
                        if (j < 1) in1 ++;
                        break;
                    }
                }
            }
        }

        double score10 = in10*1.0/allCount;
        double score5 = in5*1.0/allCount;
        double score3 = in3*1.0/allCount;
        double score1 = in1*1.0/allCount;

        System.out.println("Total: "+allCount+", Top10: "+score10+", Top5: "+score5+", Top3: "+score3+", Top1: "+score1);

        assertTrue(score10 > 0.44);
        assertTrue(score5 > 0.35);
        assertTrue(score3 > 0.28);
        assertTrue(score1 > 0.16);
    }
}
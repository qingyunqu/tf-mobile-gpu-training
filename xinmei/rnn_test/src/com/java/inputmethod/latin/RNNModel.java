package com.java.inputmethod.latin;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static java.lang.Boolean.TRUE;

public class RNNModel {
    private static final String TAG = "RNN Model";
    private long mPtr = 0;

    private final Map<String, Integer> mInDict = new HashMap<>();
    private final List<String> mOutDict = new ArrayList<>();

    private static final Pattern NUM_REGEX = Pattern.compile("[+-]*[0-9]+.*[0-9]*\\w*");
    private static final Pattern PUN_REGEX = Pattern.compile("[,;&:\\-\\$]+");

    public void destroy() {
        if(mPtr != 0) {
            nativeDestroy(mPtr);
        }
        mPtr = 0;
    }

    public boolean inited() {
        return mPtr != 0;
    }

    private void close(Closeable c) {
        if (c != null) {
            try {
                c.close();
            } catch (IOException ignored) {
            }
        }
    }

    private void initDict() throws IOException {
        FileReader in = new FileReader("res/1kw.token");
        BufferedReader br = new BufferedReader(in);

        try {
            int id = 0;
            String word;
            while ((word = br.readLine()) != null) {
                mInDict.put(word.trim(), id++);
                mOutDict.add(word.trim());
            }
        } finally {
            close(br);
        }
    }

    private void initModel() throws IOException {
        if (mPtr == 0) {
            mPtr = nativeInit("res/weights-sparse_graph-finetune-config-1kw.cfg.pb", TRUE);
        }
    }

    public void init() {
        try {
            initDict();
            initModel();
        } catch (IOException e) {
            System.err.println("fuck" + "init model failed" + e);
        }
    }

    private int convertWord2Id(String word) {
        if(mInDict.containsKey(word)) {
            return mInDict.get(word);
        } else if (mInDict.containsKey(word.toLowerCase())) {
            return mInDict.get(word.toLowerCase());
        } else if (NUM_REGEX.matcher(word).matches()) {
            return mInDict.get("<num>");
        } else if (PUN_REGEX.matcher(word).matches()) {
            return mInDict.get("<pun>");
        } else {
            return mInDict.get("<unk>");
        }
    }

    private void convertRawP2Score(int[] outputId, float[] outputP, String[] result, int[] score) {
        double sumP = 0;
        double maxP = 0;
        double[] tmpP = new double[outputP.length];
        for(int i = 0; i < outputP.length; i++) {
            double p = outputP[i];
            sumP += p;
            String word = mOutDict.get(outputId[i]);
            if(isSymbol(word)) {
                continue;
            }
            result[i] = word;
            tmpP[i] = Math.sqrt(p * (2 - p));
            maxP = maxP > tmpP[i] ? maxP : tmpP[i];
        }
        double maxScore = 150 * sumP;
        for(int i = 0; i < outputId.length; i++) {
            score[i] = result[i] != null ? Math.min(150, (int)(maxScore * tmpP[i] / maxP)) : -1;
//            System.out.println("RNN"+ "convertRawP2Score: " + result[i] + " " + score[i] + " " + outputP[i] + " : " + tmpP[i]);
        }
    }

    private boolean isSymbol(String word) {
        if(word.equals("<eos>"))
            return true;
        if(word.equals("<pad>"))
            return true;
        if(word.equals("<unk>"))
            return true;
        if(word.equals("<num>"))
            return true;
        if(word.equals("<pun>"))
            return true;
        return false;
    }

    public boolean predict() {
        if (!nativeRunBegin(mPtr)) {
            System.err.println("RNN" + "predict begin error");
            return false;
        }
        return true;
    }

    public boolean predict(String word, String[] result, int[] score) {
        if(word == null) {
            return false;
        }
        int outputSize = Math.min(
                result == null ? 0 : result.length,
                score == null ? 0 : score.length);
        int[] outputId = new int[outputSize];
        float[] outputP = new float[outputSize];
        int inputId = convertWord2Id(word);
        if (!nativeRunWord(mPtr, inputId, outputId, outputP)) {
            System.err.println("RNN" + "predict word error");
            return false;
        }
        convertRawP2Score(outputId, outputP, result, score);
        return true;
    }

    public boolean predict(List<String> words, String[] result, int[] score) {
        if(words == null || words.size() == 0) {
            return false;
        }
        if (!nativeRunBegin(mPtr)) {
            return false;
        }
        int inputSize = Math.min(3, words.size());
        int outputSize = Math.min(
                result == null ? 0 : result.length,
                score == null ? 0 : score.length);
        int[] outputId = new int[outputSize];
        float[] outputP = new float[outputSize];
        int[] inputIds = new int[inputSize];

        for(int i = 0; i < inputSize; i++) {
            inputIds[inputSize - 1 - i] = convertWord2Id(words.get(words.size() - 1 - i));
        }
        for(int i = 0; i < inputSize - 1; i++) {
            if(!nativeRunInside(mPtr, inputIds[i])) {
                System.err.println("RNN" + "predict sentence error");
                return false;
            }
        }
        if (!nativeRunWord(mPtr, inputIds[inputSize - 1], outputId, outputP)) {
            return false;
        }
        convertRawP2Score(outputId, outputP, result, score);
        return true;
    }

    private native long nativeInit(String path, Boolean testing);

    private native boolean nativeRunWord(long ptr,
                                         int inputId,
                                         int[] outputId,
                                         float[] outputP);

    private native boolean nativeRunInside(long ptr, int inputId);

    private native boolean nativeRunBegin(long ptr);

    private native void nativeDestroy(long ptr);
}

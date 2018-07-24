import com.java.inputmethod.latin.RNNModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    static {
        try {
            System.load("res/librnn_dict.so");
        } catch (Exception e) {
            System.out.println(e);
        } finally {

        }
    }

    public static void loadNativeLibrary() {
        // Ensures the static initializer is called
    }

    public static void main(String[] args) throws Exception {
        loadNativeLibrary();

        System.out.println("Hello World!");
        RNNModel model = new RNNModel();
        model.init();

        List<String> words = new ArrayList<>();
        words.add("happy");
        words.add("birthday");
        String[] results = new String[20];
        int[] socres = new int[20];
        model.predict(words, results, socres);


        System.out.println(Arrays.toString(results));
    }
}

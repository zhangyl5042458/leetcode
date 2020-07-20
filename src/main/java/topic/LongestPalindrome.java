package topic;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 17:09
 * @Description:最长回文子串
 */
public class LongestPalindrome {


    /**
     * 最长回文子串(中心扩散)
     * @param s
     * @return
     */
    public static String longestPalindrome(String s) {
        if (s == null || s.trim().length() == 0) {
            return "";
        }
        char[] c = s.toCharArray();
        int n = c.length;
        int t1, t2;
        int max = 0, start = 0, end = 0;
        for (int i = 0; i < n; i++) {
            t1 = i;
            t2 = i;
            while (t1 >= 0 && t2 < n && c[t1] == c[t2]) {
                if (t2 - t1 > max) {
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }

                t1--;
                t2++;
            }
            t1 = i;
            t2 = i + 1;
            while (t1 >= 0 && t2 < n && c[t1] == c[t2]) {
                if (t2 - t1 > max) {
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }
                t1--;
                t2++;
            }
        }
        return s.substring(start, end + 1);
    }


    /**
     * 最长回文子串(dp)
     * @param s
     * @return
     */
    public static String longestPalindromeDp(String s) {
        int length = s.length();
        if (length<2){
            return s;
        }

        char[] chars = s.toCharArray();

        int maxLen = 1;
        int begin = 0;

        boolean[][] dp = new boolean[length][length];

        //设置对角为true abc abc a=a b=b c=c
        for (int i = 0; i < length; i++) {
            dp[i][i] = true;
        }

        for (int j = 1; j < length; j++) {

            for (int i = 0; i < j; i++) {
                if (chars[i] != chars[j]){
                    dp[i][j]  = false;
                }else{
                    if (j-i<=2){
                        dp[i][j]  = true;
                    }else{
                        //取左上对角的数据  dp[i+1][j-1]之前已经有结果了

                        dp[i][j] = dp[i+1][j-1];
                    }
                }

                if (dp[i][j] && j-i+1 > maxLen){
                    maxLen = j-i+1;
                    begin = i;
                }

            }
        }
        return s.substring(begin,begin+maxLen);


    }


    /**
     * 最长回文子串(dp)
     * @param s
     * @return
     */
    public static String longestPalindromeDp1(String s) {

        int length = s.length();

        char[] chars = s.toCharArray();

        if (length<2){
            return s;
        }

        boolean[][] dp = new boolean[length][length];
        for (int i = 0; i < length; i++) {
            dp[i][i]   = true;
        }

        int begin = 0;
        int maxLen = 1;

        for (int j = 1; j < length; j++) {
            for (int i = 0; i < j; i++) {
                if (chars[i]!=chars[j]){
                    dp[i][j] = false;
                }else{
                    if (j-i<3){
                        dp[i][j] = true;
                    }else{
                        dp[i][j] = dp[i+1][j-1];
                    }
                }
                if (dp[i][j] && j-i+1 > maxLen){
                    maxLen = j-i+1;
                    begin = i;
                }
            }
        }

        return s.substring(begin,begin+maxLen);


    }


    public static String longestPalindrome1(String s) {
        if (s == null || s.trim().length() <= 0) {
            return null;
        }

        char[] c = s.toCharArray();
        int length = c.length;
        int start = 0, end = 0, max = 0;

        int t1, t2;

        for (int i = 0; i < length; i++) {
            t1 = i;
            t2 = i;
            while (t1 >= 0 && t2 < length && c[t1] == c[t2]) {
                if (t2 - t1 > max) {
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }
                t1--;
                t2++;
            }

            t1 = i;
            t2 = i + 1;
            while (t1 >= 0 && t2 < length && c[t1] == c[t2]) {
                if (t2 - t1 > max) {
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }
                t1--;
                t2++;
            }
        }
        return s.substring(start, end + 1);
    }


    /**
     * 大小为 K 且平均值大于等于阈值的子数组数目
     * <p>
     * i=0 j=i+threshold 双指针
     */
    public static int numOfSubarrays(int[] arr, int k, int threshold) {
        int result = 0;
        if (k >= arr.length) {
            return 1;
        }
        int i = 0;
        int j = i + k - 1;
        int sum = 0;
        for (int ii = 0; ii < k; ii++) {
            sum += arr[ii];
        }
        i++;
        j++;
        while (j <= arr.length - 1) {
            if (i > 0) {
                sum -= arr[i - 1];
            }
            sum += arr[j];
            if (sum / k >= threshold) {
                result++;
            }
            i++;
            j++;
        }
        return result;
    }


    /**
     * 和可被 K 整除的子数组
     *
     * @param a
     * @param k
     * @return
     */
    public static int subarraysDivByK(int[] a, int k) {

        int result = 0;
        if (a.length < 1) {
            return 0;
        }

        int sum = 0;
        for (int i : a) {
            sum += i;
        }

        int start = sum;

        int i = 0;
        int j = a.length - 1;
        while (i <= j) {
            if (i > 0) {
                sum -= a[i - 1];
            }
            if (sum % k == 0) {
                result++;
            }
            i++;
        }

        sum = start;
        i = 0;
        j = a.length - 1;
        while (i <= j) {
            if (j - 1 >0){
                sum -= a[j - 1];
            }
            if (sum % k == 0) {
                result++;
            }
            j--;
        }

        return result;

    }


    /**
     * 最长上升子序列
     * @param nums
     * @return
     */
    public static  int lengthOfLIS(int[] nums) {

        int length = nums.length;

        if (length<=0){
            return 0;
        }

        int[] dp = new int[length];
        dp[0] = 1;

        int max = 1;

        for (int j = 1; j < length; j++) {
            for (int i = 0; i < j; i++) {
                if (nums[j]>nums[i]){
                    dp[j] = Math.max(dp[i]+1,dp[j]);
                }
            }
            if (dp[j]==0){
                dp[j] = 1;
            }
            max = Math.max(max,dp[j]);
        }
        return max;
    }


    /**
     * 最长公共子序列
     * @param text1
     * @param text2
     * @return
     */
    public static int longestCommonSubsequence(String text1, String text2) {

        int length = text1.length();

        return 0;

    }


    public static void main(String[] args) {
        int[] a = {1,3,6,7,9,4,10,5,6};
        System.out.println(lengthOfLIS(a));
    }
}

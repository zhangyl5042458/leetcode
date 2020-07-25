package topic;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 17:09
 * @Description:最长回文子串
 */
public class LongestPalindrome {


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
     * 最长公共子序列
     *
     * @param text1
     * @param text2
     * @return
     */
    public static int longestCommonSubsequence(String text1, String text2) {

        if (text1 == null || text2 == null) {
            return 0;
        }

        if ("".equals(text1) || "".equals(text2)) {
            return 0;
        }

        int len1 = text1.length();
        int len2 = text2.length();
        if (len1 >= 1000 || len2 >= 1000) {
            return 0;
        }

        //后续要有i-1和j-1 dp的长度+1
        int[][] dp = new int[len1 + 1][len2 + 1];

        int max = 0;

        for (int i = 1; i < len1 + 1; i++) {
            for (int j = 1; j < len2 + 1; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    //如果值不一样，查询dp[i - 1][j],dp[i][j-1] 即使二维表的左和上的最大值
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
                max = Math.max(max, dp[i][j]);
                System.out.println("dp[" + i + "][" + j + "] ==>" + dp[i][j] + " max = " + max);
            }
        }
        return max;
    }


    /**
     * 最大子序和 easy
     * @param nums
     * @return
     */
    public static int maxSubArray(int[] nums) {

        int length = nums.length;

        int max = nums[0];

        for (int i = 1; i < length; i++) {
            //这时候的nums[i-1]代表了i-1的最大和而并不是nums[i-1]的值
            if (nums[i-1]+nums[i] > nums[i]){
                nums[i] += nums[i-1];
            }
            max = Math.max(max,nums[i]);
        }
        return max;
    }




    public static void main(String[] args) {

        List<List<Integer>> triangle = Lists.newArrayList(Lists.newArrayList());
        System.out.println(longestCommonSubsequence("abcde", "ace"));
    }
}

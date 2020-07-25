package topic;

import com.google.common.collect.Lists;

import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import datastructure.TreeNode;
import datastructure.TwoDimensionalArray;

import java.util.*;
import java.util.function.Function;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 17:09
 * @Description:最长回文子串
 */
public class LongestPalindrome {


    /**
     * 最长回文子串(中心扩散)
     *
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
     *
     * @param s
     * @return
     */
    public static String longestPalindromeDp(String s) {
        int length = s.length();
        if (length < 2) {
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
                if (chars[i] != chars[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i <= 2) {
                        dp[i][j] = true;
                    } else {
                        //取左上对角的数据  dp[i+1][j-1]之前已经有结果了

                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }

            }
        }
        return s.substring(begin, begin + maxLen);


    }


    /**
     * 最长回文子串(dp)
     *
     * @param s
     * @return
     */
    public static String longestPalindromeDp1(String s) {

        int length = s.length();

        char[] chars = s.toCharArray();

        if (length < 2) {
            return s;
        }

        boolean[][] dp = new boolean[length][length];
        for (int i = 0; i < length; i++) {
            dp[i][i] = true;
        }

        int begin = 0;
        int maxLen = 1;

        for (int j = 1; j < length; j++) {
            for (int i = 0; i < j; i++) {
                if (chars[i] != chars[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }

        return s.substring(begin, begin + maxLen);


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
            if (j - 1 > 0) {
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
     *
     * @param nums
     * @return
     */
    public static int lengthOfLIS(int[] nums) {

        int length = nums.length;

        if (length <= 0) {
            return 0;
        }

        int[] dp = new int[length];
        dp[0] = 1;

        int max = 1;

        for (int j = 1; j < length; j++) {
            for (int i = 0; i < j; i++) {
                if (nums[j] > nums[i]) {
                    dp[j] = Math.max(dp[i] + 1, dp[j]);
                }
            }
            if (dp[j] == 0) {
                dp[j] = 1;
            }
            max = Math.max(max, dp[j]);
        }
        return max;
    }



    /**
     * 整数反转
     *
     * @param x
     * @return
     */
    public static int reverse(int x) {

        boolean fushu = false;
        if (x < 0) {
            fushu = true;
            x = -x;
        }
        String s = String.valueOf(x);
        StringBuilder sb = new StringBuilder(s);
        String reStr = sb.reverse().toString();

        Integer integer = null;
        try {
            integer = Integer.valueOf(reStr);
        } catch (NumberFormatException e) {
            return 0;
        }

        if (fushu) {
            integer = -integer;
        }
        return integer;
    }


    /**
     * 字符串转换整数 (atoi)
     *
     * @param str
     * @return
     */
    public static int myAtoi(String str) {
        int i = 0, n = str.length();
        while (i < n && str.charAt(i) == ' ') {
            i++;
        }
        if (i == n) {
            return 0;
        }
        int flag = 1;
        if (str.charAt(i) == '+' || str.charAt(i) == '-') {
            if (str.charAt(i) == '-') {
                flag = -1;
            }
            i++;
        }
        int ans = 0;
        while (i < n && Character.isDigit(str.charAt(i))) {
            int temp = str.charAt(i) - '0';
            if (flag == 1 && (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && temp > 7))) {
                return Integer.MAX_VALUE;
            }

            //以正数为列，考虑稳大于和加temp才大于的情况
            if (flag == -1 && (ans > -(Integer.MIN_VALUE / 10) || (ans == -(Integer.MIN_VALUE / 10) && temp > 8))) {
                return Integer.MIN_VALUE;
            }
            ans = ans * 10 + temp;
            i++;
        }
        return ans * flag;

    }

    /**
     * 三角形最小路径和
     *
     * @param triangle
     * @return
     */
    public static int minimumTotal(List<List<Integer>> triangle) {

        int size = triangle.size();

        int[][] dp = new int[size][size];

        int min = 0;

        dp[0][0] = triangle.get(0).get(0);

        for (int i = 1; i < size; i++) {
            dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
            for (int j = 1; j < triangle.get(i).size() - 1; j++) {
                dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
            }
            dp[i][i] = dp[i - 1][i - 1] + triangle.get(i).get(i);

        }
        min = dp[size - 1][0];
        for (int i = 1; i < size; i++) {
            min = Math.min(min, dp[size - 1][i]);
        }

        return min;
    }


    /**
     * 最长回文子序列
     * dp[i][j] 标识最长序列 如果char[i]==char[j] dp[i][j] = dp[i+1][j-1]
     * 否则 j - i + 1 <= 2  dp[i][j] = 2
     * else  dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]); i向后一位 j向前一位
     *
     * @param s
     * @return
     */
    public static int longestPalindromeSubseq(String s) {

        int max = 1;

        int length = s.length();
        char[] chars = s.toCharArray();

        int[][] dp = new int[length][length];
        for (int i = 0; i < length; i++) {
            dp[i][i] = 1;
        }

        for (int j = 1; j < length; j++) {
            for (int i = j-1; i >=0; i--) {
                if (chars[i] != chars[j]) {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                } else {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                }
                max = Math.max(dp[i][j], max);
            }
        }
        return max;
    }


    public static int maxSubArray(int[] nums) {

        int length = nums.length;

        int[] dp = new int[length];

        dp[0] =nums[0];

        int max = nums[0];

        for (int i = 1; i < length; i++) {
            if (dp[i-1]+nums[i]>dp[i-1]){
                dp[i] = dp[i-1]+nums[i];
            }else{
                dp[i] = dp[i-1];
            }
            max = Math.max(max,dp[i]);
        }

        return max;
    }


    public static int maxProduct(int[] nums) {

        int length = nums.length;

        int max = nums[0];

        int[] maxDp = new int[length];
        int[] minDp = new int[length];

        maxDp[0] = nums[0];
        minDp[0] = nums[0];

        for (int i = 1; i < length; ++i) {
            maxDp[i] = Math.max(Math.max(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]),nums[i]);
            minDp[i] = Math.min(Math.min(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]),nums[i]);
            max = Math.max(max,maxDp[i]);
        }

        return max;

    }


    public static int maxProduct1(int[] nums) {

        int length = nums.length;

        int max = nums[0];

        int[] maxDp = new int[length];
        int[] minDp = new int[length];

        maxDp[0] = nums[0];
        minDp[0] = nums[0];

        for (int i = 1; i < length; ++i) {
            maxDp[i] = Math.max(Math.max(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]),nums[i]);
            minDp[i] = Math.min(Math.min(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]),nums[i]);
            max = Math.max(max,maxDp[i]);
        }

        return max;

    }


    public static int minPathSum(int[][] grid) {

        int m = grid.length;
        int n = grid[0].length;

        int[][] dp = new int[m][n];

        dp[0][0] = grid[0][0];


        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i==0 && j ==0){
                    continue;
                }
                if (i == 0) {
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j] + grid[i][j], dp[i][j - 1] + grid[i][j]);
                }
            }
        }
        return dp[m-1][n-1];
    }


    public static int findMinFibonacciNumbers(int k) {

        int cur = 1;

        List<Integer> list = new ArrayList<Integer>();
        list.add(1);
        list.add(1);

        for (int i=2;;++i){
           cur  = list.get(i-1) + list.get(i-2);
           if (cur>k){
               break;
           }else{
               list.add(cur);
           }
       }

        Integer[] ints = (Integer[]) list.toArray(new Integer[list.size()]);

        int count = 0;

        for (int i = ints.length-1; i >=0; --i) {
            if (k>=ints[i]){
                ++count;
                k -= ints[i];
            }
        }
        return count;

>>>>>>> 45535361eacf6dd33ad05eec4624973e6d6b3653
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




    public static List<List<Integer>> levelOrder(TreeNode root) {

        LinkedList<TreeNode> linkedList = new LinkedList<TreeNode>();
        LinkedList<List<Integer>> list = new LinkedList();

        if (root == null){
            return list;
        }
        linkedList.add(root);
        while (!linkedList.isEmpty()){
            List<Integer> integers = new ArrayList<Integer>();

            int size = linkedList.size();

            for (int i = 0; i < size; i++) {
                if (linkedList.size() ==0){
                    break;
                }
                integers.add(linkedList.peek().getVal());
                TreeNode node = linkedList.poll();
                if (node.getLeft()!= null){
                    linkedList.add(node.getLeft());
                }
                if (node.getRight()!= null){
                    linkedList.add(node.getRight());
                }
            }
            list.add(integers);
        }

        return list;
    }


    /**
     * 螺旋矩阵
     * @param matrix
     * @return
     */
    public static List<Integer> spiralOrder(int[][] matrix) {

        if (matrix.length<1){
            return new ArrayList<Integer>();
        }

        List<Integer> result = new ArrayList<Integer>();

        int left = 0;
        int right = matrix[0].length-1;
        int top = 0;
        int bottom = matrix.length-1;

        while (true){

            //从左往右
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }

            top++;
            if (top > bottom){
                break;
            }


            //从上到下
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }

            right--;
            if (left > right){
                break;
            }


            //从右往左
            for (int i = right; i >= left; i--) {
                result.add(matrix[bottom][i]);
            }

            bottom--;
            if (top > bottom){
                break;
            }

            //从下往上
            for (int i = bottom; i >= top; i--) {
                result.add(matrix[i][left]);
            }

            left++;
            if (left > right){
                break;
            }
        }

        return result;

    }


    /**
     * 三数之和(超时)
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSumTimeOut(int[] nums) {

        int length = nums.length;

        Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();

        List<List<Integer>> result = new ArrayList<List<Integer>>();


        for (int i = 0; i < length; i++) {
            List<Integer> list = map.getOrDefault(nums[i],new ArrayList<Integer>());
            list.add(i);
            map.put(nums[i],list);
        }

        for (int i = 0; i < length; i++) {
            int a = 0 - nums[i];
            for (int j = 0; j < length; j++){
                if (j!=i){
                    int aj = a - nums[j];
                    if (map.containsKey(aj)){
                        List<Integer> integers1 = map.get(aj);
                        ArrayList<Integer> integers2 = new ArrayList<>(integers1);
                        integers2.remove(new Integer(i));
                        if (integers2.size()>0){
                            integers2.remove(new Integer(j));
                        }
                        if (integers2.size()>0){
                            ArrayList<Integer> integers = new ArrayList<Integer>();
                            integers.add(nums[i]);
                            integers.add(nums[j]);
                            integers.add(aj);
                            integers.sort(Comparator.comparing(Integer::intValue));
                            if (!result.contains(integers)){
                                result.add(integers);
                            }
                        }
                    }
                }
            }

        }
        return result;
    }



    /**
     * 三数之和 排序+双指针
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSum(int[] nums) {

        List<List<Integer>> result = new ArrayList<List<Integer>>();

        Arrays.sort(nums);

        int length = nums.length;

        if (length <3){
            return result;
        }


        for (int i = 0; i < length; i++) {

            if (nums[i]>0){
                return result;
            }

            if (i>0 && nums[i] == nums[i-1]){
                continue;
            }

            int l = i+1;
            int r = length-1;

            while (l<r) {
                //大于0
                if (nums[i]+nums[l]+nums[r] > 0){
                    r--;
                }else if (nums[i]+nums[l]+nums[r] < 0){
                    //小于0
                    l++;
                }else{
                    //等于0
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[l]);
                    list.add(nums[r]);
                    result.add(list);

                    //一直循环到出现大于nums[l]的 既是不等于nums[l]的
                    while (l<r && nums[l] == nums[l+1]){
                        l++;
                    }
                    while (l<r && nums[r] == nums[r-1]){
                        r--;
                    }
                    l++;
                    r--;
                }
            }

        }
        return result;
    }


    /**
     * 有效的括号 栈
     * @param s
     * @return
     */
    public static boolean isValid(String s) {

        Stack<Character> stack = new Stack<>();

        char[] chars = s.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            if ("}".equals(String.valueOf(chars[i]))){
                if (stack.empty()){
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("{")) {
                    return false;
                }
            }else if ("]".equals(String.valueOf(chars[i]))){
                if (stack.empty()){
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("[")){
                    return false;
                }
            }else if (")".equals(String.valueOf(chars[i]))){
                if (stack.empty()){
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("(")){
                    return false;
                }
            }else{
                stack.push(chars[i]);
            }
        }
        if (stack.empty()){
            return true;
        }else{
            return false;
        }
    }



    public static void main(String[] args) {

        List<List<Integer>> triangle = Lists.newArrayList(Lists.newArrayList());
        System.out.println(longestCommonSubsequence("abcde", "ace"));
//        int[] a = {1, 3, 6, 7, 9, 4, 10, 5, 6};
//        List<List<Integer>> arg = Lists.newArrayList();
//        arg.add(Lists.newArrayList(1,3,1));
//        arg.add(Lists.newArrayList(1,5,1));
//        arg.add(Lists.newArrayList(4,2,1));
////        arg.add(Lists.newArrayList(4, 1, 8, 3));
//
//        arg.toArray();
//        System.out.println(minPathSum(new int[][]{{1,3,1},{1,5,1},{4,2,1}}));

//
//        TreeNode node = new TreeNode(3);
//        node.setLeft(new TreeNode(9));
//        node.setRight(new TreeNode(20));
//
//        node.getRight().setLeft(new TreeNode(15));
//        node.getRight().setRight(new TreeNode(7));
        System.out.println(isValid("()"));
    }
}

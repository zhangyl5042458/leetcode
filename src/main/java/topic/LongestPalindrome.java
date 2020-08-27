package topic;

import java.util.List;

import datastructure.ListNode;
import datastructure.TreeNode;

import java.util.*;

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
     * 大小为 K 且平均值大于等于阈值的子数组数目
     * <p>
     * i=0 j=i+threshold 双指针
     */
    public static int numOfSubarrays1(int[] arr, int k, int threshold) {
        int result = 0;
        if (k >= arr.length) {
            return 1;
        }
        int i = 0;
        int j = i + k - 1;

        int sum = 0;

        for (int i1 = 0; i1 < i + k; i1++) {
            sum += arr[i1];
        }
        if (sum / k >= threshold) {
            result++;
        }

        i++;
        j++;

        while (j <= arr.length - 1) {

            sum -= arr[i - 1];

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
            for (int i = j - 1; i >= 0; i--) {
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


    public static int maxSubArray1(int[] nums) {

        int length = nums.length;

        int[] dp = new int[length];

        dp[0] = nums[0];

        int max = nums[0];

        for (int i = 1; i < length; i++) {
            if (dp[i - 1] + nums[i] > dp[i - 1]) {
                dp[i] = dp[i - 1] + nums[i];
            } else {
                dp[i] = dp[i - 1];
            }
            max = Math.max(max, dp[i]);
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
            maxDp[i] = Math.max(Math.max(maxDp[i - 1] * nums[i], minDp[i - 1] * nums[i]), nums[i]);
            minDp[i] = Math.min(Math.min(maxDp[i - 1] * nums[i], minDp[i - 1] * nums[i]), nums[i]);
            max = Math.max(max, maxDp[i]);
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
            maxDp[i] = Math.max(Math.max(maxDp[i - 1] * nums[i], minDp[i - 1] * nums[i]), nums[i]);
            minDp[i] = Math.min(Math.min(maxDp[i - 1] * nums[i], minDp[i - 1] * nums[i]), nums[i]);
            max = Math.max(max, maxDp[i]);
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
                if (i == 0 && j == 0) {
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
        return dp[m - 1][n - 1];
    }


    public static int findMinFibonacciNumbers(int k) {

        int cur = 1;

        List<Integer> list = new ArrayList<Integer>();
        list.add(1);
        list.add(1);

        for (int i = 2; ; ++i) {
            cur = list.get(i - 1) + list.get(i - 2);
            if (cur > k) {
                break;
            } else {
                list.add(cur);
            }
        }

        Integer[] ints = (Integer[]) list.toArray(new Integer[list.size()]);

        int count = 0;

        for (int i = ints.length - 1; i >= 0; --i) {
            if (k >= ints[i]) {
                ++count;
                k -= ints[i];
            }
        }
        return count;
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
     *
     * @param nums
     * @return
     */
    public static int maxSubArray(int[] nums) {

        int length = nums.length;

        int max = nums[0];

        for (int i = 1; i < length; i++) {
            //这时候的nums[i-1]代表了i-1的最大和而并不是nums[i-1]的值
            if (nums[i - 1] + nums[i] > nums[i]) {
                nums[i] += nums[i - 1];
            }
            max = Math.max(max, nums[i]);
        }
        return max;
    }


    /**
     * 剑指 Offer 32 - II. 从上到下打印二叉树 II
     *
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrder(TreeNode root) {

        LinkedList<TreeNode> linkedList = new LinkedList<TreeNode>();
        LinkedList<List<Integer>> list = new LinkedList();

        if (root == null) {
            return list;
        }
        linkedList.add(root);
        while (!linkedList.isEmpty()) {
            List<Integer> integers = new ArrayList<Integer>();

            int size = linkedList.size();

            for (int i = 0; i < size; i++) {
                if (linkedList.size() == 0) {
                    break;
                }
                integers.add(linkedList.peek().getVal());
                TreeNode node = linkedList.poll();
                if (node.getLeft() != null) {
                    linkedList.add(node.getLeft());
                }
                if (node.getRight() != null) {
                    linkedList.add(node.getRight());
                }
            }
            list.add(integers);
        }

        return list;
    }


    /**
     * 螺旋矩阵
     *
     * @param matrix
     * @return
     */
    public static List<Integer> spiralOrder(int[][] matrix) {

        if (matrix.length < 1) {
            return new ArrayList<Integer>();
        }

        List<Integer> result = new ArrayList<Integer>();

        int left = 0;
        int right = matrix[0].length - 1;
        int top = 0;
        int bottom = matrix.length - 1;

        while (true) {

            //从左往右
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }

            top++;
            if (top > bottom) {
                break;
            }


            //从上到下
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }

            right--;
            if (left > right) {
                break;
            }


            //从右往左
            for (int i = right; i >= left; i--) {
                result.add(matrix[bottom][i]);
            }

            bottom--;
            if (top > bottom) {
                break;
            }

            //从下往上
            for (int i = bottom; i >= top; i--) {
                result.add(matrix[i][left]);
            }

            left++;
            if (left > right) {
                break;
            }
        }

        return result;

    }


    /**
     * 三数之和(超时)
     *
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSumTimeOut(int[] nums) {

        int length = nums.length;

        Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();

        List<List<Integer>> result = new ArrayList<List<Integer>>();


        for (int i = 0; i < length; i++) {
            List<Integer> list = map.getOrDefault(nums[i], new ArrayList<Integer>());
            list.add(i);
            map.put(nums[i], list);
        }

        for (int i = 0; i < length; i++) {
            int a = 0 - nums[i];
            for (int j = 0; j < length; j++) {
                if (j != i) {
                    int aj = a - nums[j];
                    if (map.containsKey(aj)) {
                        List<Integer> integers1 = map.get(aj);
                        ArrayList<Integer> integers2 = new ArrayList<>(integers1);
                        integers2.remove(new Integer(i));
                        if (integers2.size() > 0) {
                            integers2.remove(new Integer(j));
                        }
                        if (integers2.size() > 0) {
                            ArrayList<Integer> integers = new ArrayList<Integer>();
                            integers.add(nums[i]);
                            integers.add(nums[j]);
                            integers.add(aj);
                            integers.sort(Comparator.comparing(Integer::intValue));
                            if (!result.contains(integers)) {
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
     *
     * @param nums
     * @return
     */
    public static List<List<Integer>> threeSum(int[] nums) {

        List<List<Integer>> result = new ArrayList<List<Integer>>();

        Arrays.sort(nums);

        int length = nums.length;

        if (length < 3) {
            return result;
        }


        for (int i = 0; i < length; i++) {

            if (nums[i] > 0) {
                return result;
            }

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int l = i + 1;
            int r = length - 1;

            while (l < r) {
                //大于0
                if (nums[i] + nums[l] + nums[r] > 0) {
                    r--;
                } else if (nums[i] + nums[l] + nums[r] < 0) {
                    //小于0
                    l++;
                } else {
                    //等于0
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[l]);
                    list.add(nums[r]);
                    result.add(list);

                    //一直循环到出现大于nums[l]的 既是不等于nums[l]的
                    while (l < r && nums[l] == nums[l + 1]) {
                        l++;
                    }
                    while (l < r && nums[r] == nums[r - 1]) {
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
     * 四数和
     *
     * @param nums
     * @param target
     * @return
     */
    public static List<List<Integer>> fourSum(int[] nums, int target) {

        List<List<Integer>> result = new ArrayList<>();

        Arrays.sort(nums);

        int length = nums.length;

        if (length < 4) {
            return result;
        }

        for (int i = 0; i < length - 3; i++) {

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int diff = target - nums[i];

            for (int i1 = i + 1; i1 < length - 2; i1++) {

                if (i1 > i + 1 && nums[i1] == nums[i1 - 1]) {
                    continue;
                }
                int diff1 = diff - nums[i1];
                int l = i1 + 1;
                int r = length - 1;

                while (l < r) {
                    if (nums[l] + nums[r] > diff1) {
                        r--;
                    } else if (nums[l] + nums[r] < diff1) {
                        l++;
                    } else {
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[i]);
                        list.add(nums[i1]);
                        list.add(nums[l]);
                        list.add(nums[r]);
                        result.add(list);

                        //一直循环到出现大于nums[l]的 既是不等于nums[l]的
                        while (l < r && nums[l] == nums[l + 1]) {
                            l++;
                        }
                        while (l < r && nums[r] == nums[r - 1]) {
                            r--;
                        }
                        l++;
                        r--;
                    }
                }
            }
        }

        return result;
    }


    /**
     * 有效的括号 栈
     *
     * @param s
     * @return
     */
    public static boolean isValid(String s) {

        Stack<Character> stack = new Stack<>();

        char[] chars = s.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            if ("}".equals(String.valueOf(chars[i]))) {
                if (stack.empty()) {
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("{")) {
                    return false;
                }
            } else if ("]".equals(String.valueOf(chars[i]))) {
                if (stack.empty()) {
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("[")) {
                    return false;
                }
            } else if (")".equals(String.valueOf(chars[i]))) {
                if (stack.empty()) {
                    return false;
                }
                Character pop = stack.pop();
                if (!pop.toString().equals("(")) {
                    return false;
                }
            } else {
                stack.push(chars[i]);
            }
        }
        if (stack.empty()) {
            return true;
        } else {
            return false;
        }
    }


    /**
     * 最大矩形 dp
     *
     * @param matrix
     * @return
     */
    public static int maximalRectangle(char[][] matrix) {

//            int m = matrix.length;
//            int n = matrix[0].length;
//
//            int[][] dp = new int[m][n];
//
//            if (matrix[0][0] == '1'){
//                dp[0][0] = 1;
//            }
//
//            int max = dp[0][0];
//
//            for (int i = 1; i < m; i++) {
//                if (matrix[i][0] == '1'){
//                    dp[i][0] = dp[i-1][0] +1;
//                    max = Math.max(max,dp[i][0]);
//                }
//            }
//
//            for (int i = 1; i < n; i++) {
//                if (matrix[0][i] == '1'){
//                    dp[0][i] = dp[0][i-1] +1;
//                    max = Math.max(max,dp[0][i]);
//                }
//            }
//
//            for (int i = 1; i < m; i++) {
//                for (int j = 1; j < n; j++) {
//                    if (matrix[i][j] == '1'){
//                        if (matrix[i-1][j-1] != '1'){
//                            dp[i][j] = 1 + Math.max(dp[i-1][j],dp[i][j-1]);
//                        }else{
//                            dp[i][j] = 1 + dp[i-1][j]+ dp[i][j-1] - dp[i-1][j-1];
//                        }
//                        max = Math.max(max,dp[i][j]);
//                    }
//                }
//            }
//
//            return max;
        return 0;
    }

    /**
     * 最大矩形 暴力
     *
     * @param matrix
     * @return
     */
    public static int maximalRectangle1(char[][] matrix) {

        int m = matrix.length;
        if (m <= 0) {
            return 0;
        }
        int n = matrix[0].length;

        int[][] width = new int[m][n];

        int area = 0;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (j == 0) {
                        width[i][j] = 1;
                    } else {
                        width[i][j] = width[i][j - 1] + 1;
                    }
                }
                int minWidth = width[i][j];

                for (int upi = i; upi >= 0; upi--) {
                    int height = i - upi + 1;
                    minWidth = Math.min(minWidth, width[upi][j]);
                    area = Math.max(area, minWidth * height);
                }
            }
        }


        return area;
    }


    public static int findRepeatNumber(int[] nums) {

        Set<Integer> set = new HashSet<>();


        for (int num : nums) {
            if (!set.add(num)) {
                return num;
            }
        }

        return 0;
    }


    /**
     * 剑指 Offer 04. 二维数组中的查找 二分
     *
     * @param matrix
     * @param target
     * @return
     */
    public static boolean findNumberIn2DArray(int[][] matrix, int target) {

        int m = matrix.length;
        if (m <= 0) {
            return false;
        }
        int n = matrix[0].length;
        if (n <= 0) {
            return false;
        }

        for (int i = 0; i < m; i++) {
            int result = midSearch(0, n - 1, target, matrix, i);
            if (result != 0) {
                return true;
            }
        }

        return false;

    }

    private static int midSearch(int start, int end, int target, int[][] matrix, int i) {

        if (start > end) {
            return 0;
        }

        int mid = (start + end) / 2;

        if (matrix[i][mid] > target) {
            end = mid - 1;
            return midSearch(start, end, target, matrix, i);
        } else if (matrix[i][mid] < target) {
            start = mid + 1;
            return midSearch(start, end, target, matrix, i);
        } else {
            return matrix[i][mid];
        }


    }


    /**
     * 剑指 Offer 04. 二维数组中的查找 O(m+n)
     *
     * @param matrix
     * @param target
     * @return
     */
    public static boolean findNumberIn2DArray1(int[][] matrix, int target) {

        int m = matrix.length;
        if (m <= 0) {
            return false;
        }
        int n = matrix[0].length;
        if (n <= 0) {
            return false;
        }

        int i = 0;
        int j = n - 1;

        while (i < m && j >= 0) {
            if (matrix[i][j] > target) {
                j--;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                return true;
            }
        }

        return false;

    }


    /**
     * 剑指 Offer 11. 旋转数组的最小数字
     *
     * @param numbers
     * @return
     */
    public static int minArray(int[] numbers) {

        if (numbers.length < 1) {
            return 0;
        }

        if (numbers.length == 1) {
            return numbers[0];
        }

        int result = numbers[0];

        for (int i = 0; i < numbers.length - 1; i++) {
            if (numbers[i] > numbers[i + 1]) {
                return numbers[i + 1];
            }
        }

        return result;
    }


    /**
     * 剑指 Offer 12. 矩阵中的路径
     *
     * @param board
     * @param word
     * @return
     */
    public static boolean exist(char[][] board, String word) {

        int m = board.length;
        int n = board[0].length;

        if (m <= 0 || n <= 0 || word == null || "".equals(word)) {
            return false;
        }

        char[] chars = word.toCharArray();

        LinkedList<Integer> queue = new LinkedList<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == chars[0]) {
                    queue.offer(i);
                    queue.offer(j);
                }
            }
        }
        while (!queue.isEmpty()) {
            Integer i = queue.poll();
            Integer j = queue.poll();

            boolean search = search(i, j, m, n, chars, board);
            if (search) {
                return true;
            }
        }

        return false;


    }

    private static boolean search(Integer i, Integer j, int m, int n, char[] chars, char[][] board) {

        int index = 0;

        int type = 0;

        while (i < m && j < n) {

            if (index == chars.length - 1) {
                return true;
            }

            boolean find = false;

            if (type != 0) {
                //上--下
                if (type == 1) {
                    if (board[i + 1][j] == chars[index]) {
                        i++;
                        find = true;
                    }
                    if (board[i][j + 1] == chars[index]) {
                        j++;
                        type = 3;
                        find = true;
                    }
                    if (board[i][j - 1] == chars[index]) {
                        j--;
                        type = 4;
                        find = true;
                    }
                }
                //下-上
                if (type == 2) {
                    if (board[i - 1][j] == chars[index]) {
                        i--;
                        find = true;
                    }
                    if (board[i][j + 1] == chars[index]) {
                        j++;
                        type = 3;
                        find = true;
                    }
                    if (board[i][j - 1] == chars[index]) {
                        j--;
                        type = 4;
                        find = true;
                    }
                }

                //左--右
                if (type == 3) {
                    if (board[i + 1][j] == chars[index]) {
                        i++;
                        type = 1;
                        find = true;
                    }
                    if (board[i][j + 1] == chars[index]) {
                        j++;
                        find = true;
                    }
                    if (board[i - 1][j] == chars[index]) {
                        i--;
                        type = 2;
                        find = true;
                    }
                }

                //右--左
                if (type == 4) {
                    if (board[i + 1][j] == chars[index]) {
                        i++;
                        type = 1;
                        find = true;
                    }
                    if (board[i - 1][j] == chars[index]) {
                        i--;
                        type = 2;
                        find = true;
                    }
                    if (board[i][j - 1] == chars[index]) {
                        j--;
                        find = true;
                    }
                }
            } else {
                if (board[i + 1][j] == chars[index]) {
                    i++;
                    type = 1;
                    find = true;
                }
                if (i > 0 && board[i - 1][j] == chars[index]) {
                    i--;
                    type = 2;
                    find = true;
                }
                if (board[i][j + 1] == chars[index]) {
                    j++;
                    type = 3;
                    find = true;
                }
                if (j > 0 && board[i][j - 1] == chars[index]) {
                    j--;
                    type = 4;
                    find = true;
                }
            }

            if (!find) {
                return false;
            }
            index++;
        }
        return false;
    }


    /**
     * 剑指 Offer 13. 机器人的运动范围
     *
     * @param m
     * @param n
     * @param k
     * @return
     */
    public static int movingCount(int m, int n, int k) {


        int i = 0;
        int j = 0;

        if (k == 0) {
            return 1;
        }

        while (i < m && j < n) {
            if (i > 0 && i + j > k) {
                return (i + 1) * (j + 1) - (i + j - k);
            }
            if (i <= j) {
                if (i >= m) {
                    j++;
                } else {
                    i++;
                }
            } else {
                if (j >= n) {
                    i++;
                } else {
                    j++;
                }
            }
        }

        return m * n;

    }


    public static int subarraysDivByK1(int[] A, int K) {

        int result = 0;

        int length = A.length;
        if (length < 1) {
            return 0;
        }

        int sum = 0;

        for (int i = 0; i < length; i++) {
            sum += A[i];
        }

        int forsum = sum;

        for (int i = 0; i < length; i++) {
            for (int j = length-1; j > i; j--) {
                if (forsum % K == 0){
                    result++;
                }
                forsum -= A[j];
            }
            if (forsum % K ==0){
                result++;
            }
            sum -= A[i];
            forsum = sum ;
        }

        return result;

    }

    public static int subarraysDivByK2(int[] a, int k) {

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
            if (sum % k == 0) {
                result++;
            }
            sum -= a[i];
            i++;
        }

        sum = start;
        i = 0;
        j = a.length - 1;
        while (i <= j) {
            if (sum % k == 0) {
                result++;
            }
            sum -= a[j];
            j--;
        }

        return result;

    }


    public static double myPow(double x, int n) {

        return n>0?myPowMul(x,n):1.0/myPowMul(x,n);

    }

    public static double myPowMul(double x, int n) {

        if (n == 0) {
            return 1.0;
        }

        double v = myPowMul(x, n / 2);

        return n%2 ==0? v * v : v*v*x;

    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {

        //中序遍历的map  value-index
        HashMap<Integer, Integer> inMap = new HashMap<>();

        for (int i = 0; i < inorder.length; i++) {
            inMap.put(inorder[i],i);
        }

        return buildSubTree(0,0,inorder.length-1,preorder,inorder,inMap);

    }


    /**
     *
     * @param preRootIdx 前序遍历的root索引
     * @param inStart    中序的start
     * @param inEnd      中序的end
     * @param preorder   前序数据
     * @param inorder    中序数据
     * @param inMap      中序map
     * @return
     */
    private TreeNode buildSubTree(int preRootIdx, int inStart, int inEnd, int[] preorder, int[] inorder, HashMap<Integer, Integer> inMap) {

        if (inStart > inEnd){
            return null;
        }

        TreeNode root = new TreeNode(preorder[preRootIdx]);

        if (inStart == inEnd){
            return root;
        }

        //中序的索引
        Integer inIdx = inMap.get(preorder[preRootIdx]);

        root.setLeft(buildSubTree(preRootIdx+1,inStart,inIdx-1,preorder,inorder,inMap));
        //inIdx为当前中序的分割索引 inIdx-1 为左树的右边界  (inIdx-1)-inStart  = 右边界-左边界  +1 为左子树长度 +preRootIdx为前序的左子树长度索引 +1为前序右子树的第一个索引 即为右子树的root
        root.setRight(buildSubTree((preRootIdx+((inIdx-1-inStart)+1))+1,inIdx+1,inEnd,preorder,inorder,inMap));

        return root;

    }

    /**
     * 剑指 Offer 26. 树的子结构
     * @param A
     * @param B
     * @return
     */
    public static boolean isSubStructure(TreeNode A, TreeNode B) {

        if (B == null){
            return false;
        }
        int bRoot = B.getVal();

        LinkedList<TreeNode> aFindList = new LinkedList<>();
        aFindList.add(A);

        TreeNode tarNode = null;

        while(!aFindList.isEmpty()){
            TreeNode poll = aFindList.poll();
            if (poll.getVal() == bRoot){
                tarNode = poll;
                break;
            }
            aFindList.add(poll.getLeft());
            aFindList.add(poll.getRight());
        }

        if (tarNode == null ){
            return false;
        }

        LinkedList<TreeNode> bList = new LinkedList<>();
        bList.add(B);

        LinkedList<TreeNode> aList = new LinkedList<>();
        aList.add(tarNode);

        while (!bList.isEmpty()){
            TreeNode bCurr = bList.poll();
            TreeNode aCurr = aList.poll();

            if (aCurr == null && bCurr!=null){
                return false;
            }

            if (bCurr.getVal() != aCurr.getVal()){
                return false;
            }
            if (bCurr.getRight()!=null){
                bList.add(bCurr.getRight());
            }
            if (bCurr.getLeft()!=null){
                bList.add(bCurr.getLeft());
            }
            if (aCurr.getRight()!=null){
                aList.add(aCurr.getRight());
            }if (aCurr.getLeft()!=null){
                aList.add(aCurr.getLeft());
            }

        }
        return true;
    }


    public static void main(String[] args) {

//        List<List<Integer>> triangle = Lists.newArrayList(Lists.newArrayList());
//        System.out.println(longestCommonSubsequence("abcde", "ace"));
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
//        System.out.println(maximalRectangle1(new char[][]{{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','1','1','1'}}));
//        System.out.println(findRepeatNumber(new int[]{1,2,3,4,-2,1}));
//        System.out.println(findNumberIn2DArray1(new int[][]{{1,   4,  7, 11, 15},{2,   5,  8, 12, 19},{3,   6,  9, 16, 22},{10, 13, 14, 17, 24},{18, 21, 23, 26, 30}},33));
//        System.out.println(exist(new char[][]{{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}},"ABCCED"));
//        System.out.println(movingCount(3,5,4));
//        System.out.println(subarraysDivByK2(new int[]{4,5,0,-2,-3,1},5));
//        System.out.println(myPow(3,3));
        TreeNode node = new TreeNode(1);
        TreeNode node1 = new TreeNode(2);
        node1.setLeft(new TreeNode(4));
        node.setLeft(node1);
        node.setRight(new TreeNode(3));
        System.out.println(isSubStructure(node,new TreeNode(3)));
    }
}

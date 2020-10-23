package topic;

import datastructure.ListNode;

import java.util.*;

/**
 * @author: zhangyulin
 * @date: 2020-09-07 14:57
 * @description:
 */
public class Hot {


    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        if (l1 == null) {
            return l2;
        }

        if (l2 == null) {
            return l1;
        }

        int jinwei = 0;

        int first = l1.getVal() + l2.getVal();
        if (first >= 10) {
            jinwei = 1;
            first = first - 10;
        }

        ListNode dummyHead = new ListNode(0);

        ListNode firstNode = dummyHead;

        firstNode.setNext(new ListNode(first));

        firstNode = firstNode.getNext();

        l1 = l1.getNext();
        l2 = l2.getNext();

        while (l1 != null || l2 != null) {
            int l1Val = 0;
            int l2Val = 0;
            if (l1 != null) {
                l1Val = l1.getVal();
            }
            if (l2 != null) {
                l2Val = l2.getVal();
            }
            int sum = l1Val + l2Val + jinwei;
            if (sum >= 10) {
                jinwei = 1;
                sum = sum - 10;
            } else {
                jinwei = 0;
            }
            ListNode cur = new ListNode(sum);
            firstNode.setNext(cur);
            firstNode = firstNode.getNext();

            if (l1 != null) {
                l1 = l1.getNext();
            }
            if (l2 != null) {
                l2 = l2.getNext();
            }
        }

        return dummyHead.getNext();

    }


    public static int[] twoSum(int[] nums, int target) {

        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }

        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff) || map.get(diff) != i) {
                return new int[]{i, map.get(diff)};
            }
        }

        return null;

    }


    public static int lengthOfLongestSubstring(String s) {

        HashSet<Character> set = new HashSet<>();

        char[] chars = s.toCharArray();

        int max = 0;

        int j = 1;

        for (int i = 0; i < chars.length; i++) {
            if (i != 0) {
                set.remove(chars[i - 1]);
            }
            set.add(chars[i]);

            if (j == i) {
                ++j;
            }


            while (j < chars.length && !set.contains(chars[j])) {
                set.add(chars[j]);
                ++j;
            }

            max = Math.max(max, (j - 1) - i + 1);

        }

        return max;

    }


    public static int lengthOfLongestSubstring1(String s) {

        Map<Character, Integer> map = new HashMap<>();

        char[] chars = s.toCharArray();

        int max = 0;

        for (int i = 0; i < chars.length; i++) {

            if (map.containsKey(chars[i])) {
                Integer pre = map.get(chars[i]);

                max = Math.max(max, i - pre);

                map.put(chars[i], i);
            } else {
                map.put(chars[i], i);
                ++max;
            }

        }

        return max;

    }


    public static class LengthOfLongestSubstringClass {

        public LengthOfLongestSubstringClass(Character character, Integer index) {
            this.character = character;
            this.index = index;
        }

        private Character character;
        private Integer index;
    }

    public static int longestValidParentheses(String s) {

        int result = 0;

        int max = 0;

        Stack<A> stack = new Stack<>();

        char[] chars = s.toCharArray();

        for (int i = 0; i < chars.length; i++) {
            if (")".equals(String.valueOf(chars[i]))) {
                if (stack.empty()) {
                    max = Math.max(max, longestValidParentheses(s.substring(i + 1)));
                    break;
                }
                stack.pop();
                result += 2;
            } else {
                stack.push(new A(chars[i], i));
            }
        }

        while (!stack.empty()) {
            max = stack.pop().index;
        }

        max = Math.max(max, result);

        return max;
    }

    public static class A {
        private Character val;
        private Integer index;

        public A(Character val, Integer index) {
            this.val = val;
            this.index = index;
        }
    }


    public static int longestValidParenthesesDp(String s) {

        int max = 0;

        char[] chars = s.toCharArray();

        int[] dp = new int[s.length()];

        for (int i = 0; i < dp.length; i++) {
            dp[i] = 0;
        }

        for (int i = 0; i < chars.length; i++) {
            if (')' == chars[i]) {
                if (i == 0) {
                    continue;
                }
                if (chars[i - 1] == '(') {
                    //如果之前的字符是( 两者形成长度为2的括号,dp[i] = dp[i-2]+2;
                    if (i >= 2) {
                        dp[i] = dp[i - 2] + 2;
                    }
                } else {
                    //如果之前的字符是 ) 需要找与之对应的( 比如 (()) 第3个index, 3-dp[3-1]-1 是第一个 ( 所以与之对应的index是 i-dp[i-1]-1
                    if (chars[i - dp[i - 1] - 1] == '(' && i - dp[i - 1] - 1 >= 0) {
                        dp[i] = dp[i - 1] + 2;
                        //还需要加上 i-dp[i-1]-1 之前的dp
                        if (i - dp[i - 1] - 2 >= 0) {
                            dp[i] = dp[i] + dp[i - dp[i - 1] - 2];
                        }
                    }

                }
            }
            max = Math.max(max, dp[i]);
        }

        return max;
    }

    public static List<List<Integer>> combine(int n, int k) {

        List<List<Integer>> result = new ArrayList<>();

        if (n < 0) {
            return null;
        }

        if (k <= 0) {
            return null;
        }

        if (n < k) {
            k = n;
        }

        List<Integer> arr = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            arr.add(i + 1);
        }

//        for (Integer integer : arr) {
//
//        }
//
//        for (int i = 0; i < arr.size(); i++) {
//            List<List<Integer>> sub = getSub(arr.get(i), arr.subList(i+1,arr.size()));
//        }
        return getSub(null, arr);
    }


    public static List<List<Integer>> getSub(Integer parent, List<Integer> arr) {
        if (arr.size() == 0) {
            return new ArrayList<>();
        }

        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < arr.size(); i++) {
            List<List<Integer>> sub = getSub(arr.get(i), arr.subList(i + 1, arr.size()));
            if (parent != null) {
                for (List<Integer> integers : sub) {
                    integers.add(0, parent);
                }
            }
            result.addAll(sub);
        }

        return result;

//            List<List<Integer>> result  = new ArrayList<>();
//            for (Integer integer : arr) {
//                List<Integer> inList = new ArrayList<>();
//                inList.add(integer);
//                result.add(inList);
//            }
//            return result;
//
//        Integer integer = arr.get(0);
//        arr.remove(0);
//        List<List<Integer>> sub = getSub(arr, k - 1);
//
//        if (sub!=null){
//            sub.forEach(i->{
//                i.add(0,integer);
//            });
//        }
//        return sub;
    }


    public static void nextPermutation(int[] nums) {

        int len = nums.length;

        for (int i = len - 1; i > 0; --i) {
            if (nums[i] > nums[i - 1]) {
                int j = i;
                while (j < len) {
                    if (nums[j] > nums[i - 1]) {
                        ++j;
                    } else {
                        --j;
                        break;
                    }
                }
                if (j >= len) {
                    j = len - 1;
                }
                exchange(i - 1, j, nums);
                int i1 = i;
                int j1 = len - 1;
                while (i1 < j1) {
                    exchange(i1, j1, nums);
                    ++i1;
                    --j1;
                }
                return;
            }
        }

        int i = 0;
        int j = len - 1;

        while (i < j) {
            exchange(i, j, nums);
            ++i;
            --j;
        }

    }


    public static void exchange(int i, int j, int[] nums) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    /**
     * 删除排序数组中的重复项
     */
    public static int removeDuplicates(int[] nums) {

        if (nums.length < 2) {
            return 1;
        }

        int i = 0;


        for (int j = 1; j < nums.length; j++) {
            if (nums[i] != nums[j]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;

    }


    /**
     * 长度最小的子数组  滑动窗口
     *
     * @param s
     * @param a
     * @return
     */
    public static int minSubArrayLen(int s, int[] a) {

        int min = Integer.MAX_VALUE;

        int sum = 0;

        int i = 0;
        for (int j = 0; j < a.length; j++) {
            sum += a[j];
            while (sum >= s) {
                int length = j - i + 1;
                min = Math.min(min, length);
                sum -= a[i];
                i++;
            }
        }
        return min;

    }


    /**
     * 全排列 回溯
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();

        int length = nums.length;
        if (length <= 0) {
            return res;
        }

        Deque<Integer> path = new ArrayDeque<Integer>();
        boolean[] used = new boolean[length];
        dfs(nums, length, 0, path, used, res);

        return res;

    }

    /**
     * dfs
     *
     * @param nums   数组
     * @param length 数组长度
     * @param depth  dfs的深度
     * @param path   路径
     * @param used   是否已经被用了
     * @param res    结果集
     */
    private void dfs(int[] nums, int length, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {

        if (depth == length) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            path.addLast(nums[i]);
            dfs(nums, length, depth + 1, path, used, res);
            //回溯的关键代码，将i回溯，用过的设为没有用，并从栈中移除
            used[i] = false;
            path.removeLast();
        }


    }


    public static int[] searchRange(int[] nums, int target) {

        int length = nums.length;

        if (length < 0) {
            return null;
        }

        if (length == 1){
            if (nums[0] == target){
                return new int[]{0,0};
            }
        }


        Integer min = Integer.MAX_VALUE;
        Integer max = Integer.MIN_VALUE;
        return searchRange(nums, target, 0, length - 1, min, max);

//        if (min!= Integer.MAX_VALUE && max != Integer.MIN_VALUE){
//            return new int[]{min,max};
//        }
//        return new int[]{-1,-1};

    }

    private static int[] searchRange(int[] nums, int target, int start, int end, Integer min, Integer max) {

        if (start < 0 || end > nums.length - 1) {
            return null;
        }

        if (end < start) {
            return new int[]{-1, -1};
        }

        int index = (end + start) / 2;

        if (nums[index] == target) {
            min = Math.min(min, index);
            max = Math.max(max, index);


            int forwardIndex = index - 1;

            while (forwardIndex >=0 && nums[forwardIndex] == target) {
                min = Math.min(min, forwardIndex);
                --forwardIndex;
            }

            int backwardIndex = index + 1;
            while (backwardIndex <= nums.length-1 &&nums[backwardIndex] == target) {
                max = Math.max(max, backwardIndex);
                ++backwardIndex;
            }

            return new int[]{min, max};
        } else if (nums[index] > target) {
            return searchRange(nums, target, start, index - 1, min, max);
        } else {
            return searchRange(nums, target, index + 1, end, min, max);
        }
    }

    private static void findSame(int[] nums, int target, int index, boolean forward, boolean backward, Integer min, Integer max) {

        if (forward) {
            while (nums[index] == target) {
                min = Math.min(min, index);
                --index;
            }
        }

        if (backward) {
            while (nums[index] == target) {
                max = Math.max(max, index);
                ++index;
            }
        }

    }


    /**
     * 组合总和 (回溯)
     * @param candidates
     * @param target
     * @return
     */
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> res = new ArrayList<>();

        Deque<Integer> path = new ArrayDeque<Integer>();

        dfs(candidates,target,0,path,res);

        return res;

    }

    private static void dfs(int[] candidates, int target, int index, Deque<Integer> path, List<List<Integer>> res) {

        if (target ==0) {
            res.add(new ArrayList<>(path));
            return;
        }

        if (index == candidates.length){
            return;
        }

        dfs(candidates,target,index+1,path,res);

        if (target - candidates[index] >= 0){
            path.addLast(candidates[index]);
            dfs(candidates,target-candidates[index],index,path,res);
            path.removeLast();
        }
    }


    public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> combine = new ArrayList<Integer>();
        dfs2(candidates, target, ans, combine, 0);
        return ans;
    }

    public static void dfs2(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            ans.add(new ArrayList<Integer>(combine));
            return;
        }

        for (int i = idx; i < candidates.length; i++) {
//            if (target - candidates[i] >= 0) {
                combine.add(candidates[i]);
                dfs2(candidates, target - candidates[i], ans, combine, i+1);
                combine.remove(combine.size() - 1);
//            }
        }

//
//        // 直接跳过
//        dfs2(candidates, target, ans, combine, idx + 1);
//        // 选择当前数
//        if (target - candidates[idx] >= 0) {
//            combine.add(candidates[idx]);
//            dfs2(candidates, target - candidates[idx], ans, combine, idx);
//            combine.remove(combine.size() - 1);
//        }
    }




    /**
     * 组合总和 (回溯+剪枝) 使用排序+target的大小判断
     * @param candidates
     * @param target
     * @return
     */
    public static List<List<Integer>> combinationSum1(int[] candidates, int target) {

        List<List<Integer>> res = new ArrayList<>();

        Deque<Integer> path = new ArrayDeque<Integer>();

        Arrays.sort(candidates);

        dfs1(candidates,target,0,0,path,res);

        return res;

    }

    private static void dfs1(int[] candidates, int target, int sum, int curIndex, Deque<Integer> path, List<List<Integer>> res) {

        if (target ==sum) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = curIndex; i < candidates.length; i++) {
            if (sum + candidates[i] <=target){
                path.addLast(candidates[i]);
                dfs1(candidates,target,sum+candidates[i],i,path,res);
                path.removeLast();
            }else{
                break;
            }
        }
    }


    public static int trap(int[] height) {

        int length = height.length;

        int leftMax = 0;
        int rightMax = 0;

        int left = 0;
        int right = length-1;

        int res = 0;

        while (left < right){

            if (height[left] < height[right]){
                if (height[left] > leftMax){
                    leftMax = height[left];
                }else{
                    res += (leftMax - height[left]);
                }
                left++;
            }else{
                if (height[right] > rightMax){
                    rightMax = height[right];
                }else{
                    res += (rightMax-height[right]);
                }
                --right;
            }

        }

        return res;

    }



    public static List<String> generateParenthesis(int n) {

        List<String> res = new ArrayList<>();


        if (n==0){
            return res;
        }

        dfsgenerateParenthesis("",0,0,n,res);

        return res;

    }

    private static void dfsgenerateParenthesis(String str, int left, int right, int n, List<String> res) {
        if (str.length() == n*2){
            res.add(str);
            return;
        }


        if(left<n){
            String temp = str;
            str += "(";
            left+=1;
            dfsgenerateParenthesis(str,left,right,n,res);
            str = temp;
        }

        if (right<left){
            String temp = str;
            str += ")";
            right+=1;
            dfsgenerateParenthesis(str,left,right,n,res);
            str = temp;
        }



    }


    public static int longestConsecutive(int[] nums) {

        int max = 0;

        HashSet<Integer> set = new HashSet<Integer>();
        for (int num : nums) {
            set.add(num);
        }

        for (int num : nums) {
            int len = 1;
            while (set.contains(num-1)){
                ++len;
                --num;
            }
            max = Math.max(max,len);
        }

        return max;


    }


    public static int maximalSquare(char[][] matrix) {

        if (matrix.length<1){
            return 0;
        }
        if (matrix[0].length<1){
            return 0;
        }

        int max = 0;

        int[][] dp = new int[matrix.length][matrix[0].length];

        if (matrix[0][0] == '1'){
            dp[0][0] = 1;
            max = 1;
        }else{
            dp[0][0] = 0;
        }



        for (int i = 1; i < matrix.length; i++) {
            if (matrix[i][0] == '1'){
                dp[i][0] = 1;
            }
            max = Math.max(max,dp[i][0]);
        }


        for (int i = 1; i < matrix[0].length; i++) {
            if (matrix[0][i] == '1'){
                dp[0][i] = 1;
            }
            max = Math.max(max,dp[0][i]);
        }

        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1'){
                    dp[i][j] = Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]))+1;
                }else{
                    dp[i][j] = 0;
                }
                max = Math.max(max,dp[i][j]);
            }
        }

        return max*max;
    }



    public static int maxArea(int[] height) {
        int len = height.length;
        if (len<2){
            return 0;
        }


        int area = 0;

        boolean flag = false;

        for (int i = 0; i < len; i++) {
            for (int j = i+1; j < len; j++) {
                area = Math.max(area, (j-i)*Math.min(height[i], height[j]));
            }
        }

        return area;

    }


    public static int search(int[] nums, int target) {
        int length = nums.length;



        int i = 0;
        int j = length-1;


        while (i<j){
            int mid = (j-1)/2;

            if (nums[mid]== target){
                return mid;
            }else if (nums[i]<nums[mid-1]){
                //左边有序
                if (target>=nums[i] && target <=nums[mid-1]){
                    j = mid-1;
                }else{
                    i = mid+1;
                }
            }else{
                if (target>=nums[mid+1] && target <=nums[j]){
                    i = mid+1;
                }else{
                    j = mid-1;
                }

            }

        }

        return -1;

    }



    public static int maxProfit(int[] prices) {

        int res = 0;

        int length = prices.length;

        if (length<=0){
            return 0;
        }
        int min = Integer.MAX_VALUE;

        for (int i = 0; i < length; i++) {
            if (prices[i]< min){
                min = prices[i];
            }else if (prices[i] - min > res){
                res = prices[i] - min;
            }
        }
        return res;

    }


    public static int climbStairs(int n) {

        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];

    }





    public static int searchMid(int[] nums,int start,int end,int target){

        if (start == end){
            if (nums[start] == target){
                return start;
            }else{
                return -1;
            }
        }

        int mid = (end - start) / 2;

        if (target == nums[mid]){
            return mid;
        }else if (target > nums[mid]){
            return searchMid(nums,mid+1,end,target);
        }else{
            return searchMid(nums,start,mid-1,target);
        }

    }


//    public static int largestRectangleArea(int[] heights) {
//
//        int max = 0;
//
//        for (int i = 0; i < heights.length; i++) {
//            for (int j = i+1; j < heights.length; j++) {
//
//            }
//        }
//    }



    public static String decodeString1(String s) {

        StringBuilder res = new StringBuilder();


        Deque<String> stack = new ArrayDeque<>();

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (String.valueOf(c).matches("[0-9]+")){
                StringBuilder sb = new StringBuilder(String.valueOf(c));
                //数字放入栈
                for (int j = i+1; j < s.length(); j++) {
                    if (String.valueOf(s.charAt(j)).matches("[0-9]+")){
                        sb.append(s.charAt(j));
                    }else{
                        i=j-1;
                        break;
                    }
                }
                stack.push(sb.toString());
            }else if (c==']'){
                StringBuilder sb = new StringBuilder();
                for(;;){
                    String popStr = stack.pop();
                    if (!popStr.equals("[")){
                        sb.append(popStr);
                    }else{
                        break;
                    }
                }
                StringBuilder reverseStr = sb.reverse();

                String popInteger = stack.pop();
                Integer times = Integer.valueOf(popInteger);
                for (int integer = 0; integer < times; integer++) {
                    stack.push(reverseStr.toString());
                }
            }else{
                stack.push(String.valueOf(c));
            }
        }

        while (!stack.isEmpty()){
            String pop = stack.pop();
            res.append(new StringBuffer(pop).reverse());
        }
        return res.reverse().toString();
    }



    public int findDuplicate(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            for (int j = i+1; j < len; j++) {
                if (nums[i] == nums[j]){
                    return nums[i];
                }
            }
        }
        return 0;
    }



    public static int subarraySum(int[] nums, int k) {

        int res = 0;

        int length = nums.length;
        if (length<1){
            return 0;
        }
        int sum = nums[0];
        if (sum == k){
            res++;
        }

        int left = 0;

        for (int i = 1; i < length; i++) {
            sum += nums[i];
            if (sum==k){
                res++;
            }else if (sum>k){
                while (left<=i){
                    sum -= nums[left];
                    ++left;
                    if (sum < k){
                        break;
                    }
                    if (sum == k){
                        res++;
                    }

                }
            }
        }
        return res;
    }


    public static int lengthOfLongestSubstring11(String s) {
        Map<Character,Integer> map = new HashMap<>();

        int result = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!map.containsKey(c)){
                map.put(c,i);
            }else{
                result = Math.max(result,map.size());
                Integer integer = map.get(c);
                map.keySet().removeIf(character -> map.get(character) < integer);
                map.put(c,i);
            }
        }

        return Math.max(result,map.size());
    }


    public static int lengthOfLongestSubstring12(String s) {
        Set<Character> set = new HashSet<>();

        int result = 0;
        int j = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);

            if (i!=0){
                set.remove(c);
            }

            while (j<s.length() && !set.contains(s.charAt(j))){
                set.add(s.charAt(j));
                ++j;
            }
            result = Math.max(result,j-i);

        }

        return result;
    }




    public static String longestPalindrome123(String s) {

        int length = s.length();
        boolean[][] dp = new boolean[length][length];

        for (int i = 0; i < length; i++) {
            dp[i][i] = true;
        }

        int result = 1;

        String resultStr = String.valueOf(s.charAt(0));

        for (int i = 0; i < length; i++) {
            for (int j = 0; j < i; j++) {
                if (s.charAt(j) == s.charAt(i)){
                    if (i-j+1 > 2){
                        if (dp[j+1][i-1]){
                            dp[j][i] = true;
                            if (i-j+1 > result){
                                resultStr = s.substring(j,i+1);
                            }
                            result = Math.max(result,i-j+1);
                        }
                    }else{
                        dp[j][i] = true;
                        if (i-j+1 > result){
                            resultStr = s.substring(j,i+1);
                        }
                        result = Math.max(result,i-j+1);
                    }
                }
            }
        }

        return resultStr;
    }


    public List<List<Integer>> permute123(int[] nums) {

        int length = nums.length;

        List<List<Integer>> result = new ArrayList<>();

        boolean[] used = new boolean[length];

        Deque<Integer> stack = new ArrayDeque<>();

        dfs123(nums,used,stack,result,length);

        return result;

    }

    private void dfs123(int[] nums, boolean[] used, Deque<Integer> stack, List<List<Integer>> result, int length) {

        if (stack.size() == length){
            result.add(new ArrayList<>(stack));
        }

        for (int i = 0; i < length; i++) {
            if (used[i]){
                continue;
            }
            used[i] = true;
            stack.addLast(nums[i]);
            dfs123(nums,used,stack,result,length);
            used[i] = false;
            stack.removeLast();
        }

    }


    public static boolean exist(char[][] board, String word) {

        int length = word.length();

        char first = word.charAt(0);

        boolean result = false;

        List<int[]> list = getFirst(board,first);

        if (list.size()==0){
            return false;
        }else{
            if (list.size()==1 && word.length() == 1){
                return true;
            }
        }

        for (int[] ints : list) {
            int i = ints[0];
            int j = ints[1];
            boolean[][] used = new boolean[board.length][board[0].length];
            result = dfsexist(board,ints,0,used,length,word);
            if (result){
                return result;
            }
        }
        return result;

    }

    private static boolean dfsexist(char[][] board, int[] ints, int index, boolean[][] used, int length, String word) {

        if (index >length-1){
            return true;
        }

        int i = ints[0];
        int j = ints[1];

        if (used[i][j]){
            return false;
        }
        int high = board.length;
        int wide = board[0].length;

        if (board[i][j] == word.charAt(index)){
            used[i][j] = true;
            //上
            if (i-1>=0){
                boolean dfsexist = dfsexist(board, new int[]{i - 1, j}, index + 1, used, length, word);
                if (dfsexist){
                    return dfsexist;
                }
            }


            //下
            if (i+1<=high-1){
                boolean dfsexist = dfsexist(board, new int[]{i + 1, j}, index + 1, used, length, word);
                if (dfsexist){
                    return dfsexist;
                }
            }

            //左
            if (j-1>=0){
                boolean dfsexist = dfsexist(board, new int[]{i, j - 1}, index + 1, used, length, word);
                if (dfsexist){
                    return dfsexist;
                }
            }


            //右
            if (j+1<=wide-1){
                boolean dfsexist = dfsexist(board, new int[]{i , j + 1}, index + 1, used, length, word);
                if (dfsexist){
                    return dfsexist;
                }
            }
            used[i][j] = false;
        }else{
            return false;
        }
        return false;

    }

    private static List<int[]> getFirst(char[][] board, char first) {
        List<int[]> result = new ArrayList<>();
        int high = board.length;
        int length = board[0].length;
        for (int i = 0; i < high; i++) {
            for (int j = 0; j < length; j++) {
                if (board[i][j] == first) {
                    result.add(new int[]{i,j});
                }
            }
        }
        return result;
    }




    public static int maxSubArray123(int[] nums) {

        int[] dp = new int[nums.length];
        dp[0] = nums[0];

        int result = nums[0];

        for (int i = 1; i < nums.length; i++) {
            dp[i] = dp[i-1]>=0? nums[i]+dp[i-1]:nums[i];
            result = Math.max(result,dp[i]);
        }

        return result;
    }


    /**
     * 暴力
     * @param height
     * @return
     */
    public static int trap1(int[] height) {

        int res = 0;

        for (int i = 0; i < height.length; i++) {
            int lmax = 0;
            int rmax = 0;
            for (int l = 0; l <= i; l++) {
                lmax = Math.max(lmax,height[l]);
            }
            for (int r = i; r <= height.length-1; r++) {
                rmax = Math.max(rmax,height[r]);
            }
            res += Math.min(lmax,rmax)-height[i];
        }

        return res;
    }


    /**
     * 动态规划
     * @param height
     * @return
     */
    public static int trap2(int[] height) {

        if (height.length==0){
            return 0;
        }

        int[] lmaxDp = new int[height.length];
        int[] rmaxDp = new int[height.length];
        lmaxDp[0] = height[0];
        rmaxDp[height.length-1] = height[height.length-1];

        for (int i = 1; i < height.length; i++) {
            lmaxDp[i] = Math.max(height[i],lmaxDp[i-1]);
        }

        for (int i = height.length-2; i >=0; i--) {
            rmaxDp[i] = Math.max(height[i],rmaxDp[i+1]);
        }

        int res = 0;

        for (int i = 0; i < height.length; i++) {
            res += Math.min(lmaxDp[i],rmaxDp[i])-height[i];
        }
        return res;
    }


    /**
     * 双指针
     * @param height
     * @return
     */
    public static int trap3(int[] height) {

        int res = 0;

        int l = 0;
        int r = height.length-1;

        int lmax = 0;
        int rmax = 0;

        while (l<r){
            if (height[l]<height[r]){
                if (height[l]<lmax){
                    res += lmax-height[l];
                }else{
                    lmax = height[l];
                }
                ++l;
            }else{
                if (height[r]<rmax){
                    res += rmax - height[r];
                }else{
                    rmax = height[r];
                }
                --r;
            }
        }
        return res;
    }


    public static int lengthOfLongestSubstring1234(String s) {

        if (s.length() == 1){
            return 1;
        }
        Map<Character, Integer> map = new HashMap<>();
        int res = 0;
        int tmp = 0;
        for (int i = 0; i < s.length(); i++) {
            Integer canFindIndex = map.getOrDefault(s.charAt(i), -1);
            tmp = tmp < i - canFindIndex ? tmp + 1 : i - canFindIndex;
            map.put(s.charAt(i),i);
            res = Math.max(res,tmp);
        }
        return res;
    }


    public static int[][] rotate(int[][] matrix) {
        int n = matrix.length;
        int[][] oldVal = new int[n][n];
        boolean[][] used = new boolean[n][n];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                oldVal[i][j]= used[i][j]?oldVal[i][j]:matrix[i][j];
                int[] rval = getRotateValue(i,j,n);
                oldVal[rval[0]][rval[1]] = matrix[rval[0]][rval[1]];
                matrix[rval[0]][rval[1]] = oldVal[i][j];
                used[rval[0]][rval[1]] = true;
            }
        }
        return matrix;
    }

    private static int[] getRotateValue(int i, int j, int n) {
        return new int[]{j,n-1-i};
    }


    public static double myPow(double x, int n) {
        double res = myPow1233(x,n);
        return n>0?res:1/res;
    }

    private static double myPow1233(double x, int n) {
        if (n==0){
            return 1;
        }

        double v = myPow1233(x, n / 2);

        return n%2==0?v*v:v*v*x;
    }


//    public boolean isEscapePossible(int[][] blocked, int[] source, int[] target) {
//
//        HashSet<int[]> set = new HashSet<>();
//        for (int i = 0; i < blocked.length; i++) {
//            for (int j = 0; j < blocked[0].length; j++) {
//                set.add(new int[]{i,j});
//            }
//        }
//
//        int i = target[0];
//        int j = target[1];
//
//
//
//
//
//    }


    public static int[] sortedSquares(int[] A) {

        int length = A.length;

        int[] res = new int[length];

        int i = 0;
        int j = length-1;

        for (int index = j; index >=0; --index) {
            int jval = A[j] * A[j];
            int ival = A[i] * A[i];
            if (ival>=jval){
                res[index] = ival;
                i++;
            }else{
                res[index] = jval;
                --j;
            }
        }
        return res;
    }


    public static List<String> commonChars(String[] A) {

        String first = A[0];
        Map<Character,Integer> map = new HashMap();

        for (int i = 0; i < A[0].length(); i++) {
            char c = first.charAt(i);
            Integer integer = map.get(c);
            if (integer == null){
                map.put(c,0);
            }else{
                map.put(c,integer+1);
            }
        }

        for (int i = 1; i < A.length; i++) {
            for (int j = 0; j < A[i].length(); j++) {
                if (map.containsKey(A[i].charAt(j))){
                    int count = 1;
                    for (int k = 0; k < A[i].length(); k++) {
                        if (k==j){
                            continue;
                        }
                        if (map.containsKey(A[i].charAt(k))){
                            ++count;
                        }
                    }
                    map.put(A[i].charAt(j),count);
                }
            }
        }

        List<String> res  = new ArrayList<>();

        map.keySet().forEach(i->{
            if (map.get(i)!=0){
                res.add(i.toString());
            }
        });

        return res;
    }




    public static int findKthLargest(int[] nums, int k) {

        return quickSort(nums,0,nums.length-1,nums.length-k);

    }

    private static int quickSort(int[] nums, int low, int high, int index) {

        int mid = getMid(nums,low,high);

        if (mid == index){
            return nums[mid];
        }else if (mid>index){
            return quickSort(nums, low,mid-1,index);
        }else{
            return quickSort(nums, mid+1,high,index);
        }


    }

    private static int getMid(int[] nums, int low, int high) {

        int target = nums[low];

        while (low < high){

            while (low < high && nums[high] >= target){
                --high;
            }
            int highval = nums[high];
            nums[high] = nums[low];
            nums[low] = highval;



            while (low < high && nums[low] <= target){
                ++low;
            }
            int lowval = nums[low];
            nums[low] = nums[high];
            nums[high] = lowval;
        }

        return low;
    }



    public static List<String> generateParenthesis123(int n) {


        List<String> ans = new ArrayList<>();


        dfs111(new StringBuilder(),0,n,0,0,ans);

        return ans;

    }

    private static void dfs111(StringBuilder first, int currentLen, int n,int zuo,int you, List<String> ans) {

        if (currentLen == n*2){
            ans.add(first.toString());
            return;
        }

        if (zuo < n){
            first.append("(");
            dfs111(first,currentLen+1,n,zuo+1,you,ans);
            first.deleteCharAt(first.length()-1);
        }


        if (you < zuo){
            first.append(")");
            dfs111(first,currentLen+1,n,zuo,you+1,ans);
            first.deleteCharAt(first.length()-1);
        }
    }


    public static boolean isLongPressedName(String name, String typed) {

        int length = typed.length();
        int i = 0;
        int j = 0;
        while (i < length - 1) {
            if (name.charAt(j) != typed.charAt(i)) {
                return false;
            }
            int len = 1;
            int jlen = 1;
            boolean has = false;
            while (typed.charAt(i) == typed.charAt(i + 1)) {
                ++i;
                ++len;
            }
            while (name.charAt(j) == name.charAt(j + 1)) {
                ++j;
                ++jlen;
                has = true;
            }
            if (jlen > len) {
                return false;
            }
            if (!has) {
                ++j;
            }
        }

        return true;
    }


    public static void sortColors(int[] nums) {

        int[] res = new int[nums.length];

        List<Integer> list = new ArrayList<>();

        int j  = 0;
        int  k = res.length-1;
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (num==0){
                res[j] = num;
                ++j;

            }else if (num==2){
                res[k] = num;
                --k;
            }else{
                list.add(num);
            }
        }

        for (Integer integer : list) {
            res[j] = integer;
            ++j;
        }

        nums = res;

    }



    public static List<Integer> partitionLabels(String S) {
        if (S==null){
            return null;
        }
        if (S.equals("")){
            return null;
        }
        List<Integer> res = new ArrayList<>();
        char temp = S.charAt(0);

        int index = 0;
        while (index<S.length()){
            int start = S.indexOf(temp);
            int i = S.lastIndexOf(temp);
            for (int i1 = start; i1 < i; i1++) {
                i = Math.max(i,S.lastIndexOf(S.charAt(i1)));
            }
            res.add(i-start+1);
            index = i;
            if (i==S.length()-1){
                break;
            }
            temp = S.charAt(i+1);
        }

        return res;

    }


    public String[] permutation(String s) {

        int length = s.length();

        char[] chars = s.toCharArray();

        Arrays.sort(chars);

        boolean[] used = new boolean[length];

        List<String> dataList = new ArrayList<>();

        permutationDfs(chars,length,0,new StringBuilder(),dataList,used,new HashSet<String>());

        String[] res = new String[dataList.size()];

        for (int i = 0; i < dataList.size(); i++) {
            res[i] = dataList.get(i);
        }

        return res;

    }

    private void permutationDfs(char[] s, int length, int path, StringBuilder cur, List<String> dataList, boolean[] used, HashSet<String> set) {

        if (path==length){
            if (!set.contains(cur.toString())){
                dataList.add(cur.toString());
            }
            return;
        }

        for (int i1 = 0; i1 < length; i1++) {
            if (used[i1]){
                continue;
            }
            if (i1>0 && s[i1]==s[i1-1] && !used[i1-1]){
                continue;
            }
            cur.append(s[i1]);
            used[i1] = true;
            permutationDfs(s,length,path+1,cur,dataList, used, set);
            used[i1] = false;
            cur.deleteCharAt(cur.length()-1);
        }




    }



    public static List<List<Integer>> subsetsWithDup(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();

        subsetsWithDupDfs(nums,new ArrayList<>(),0,nums.length,res,new boolean[nums.length]);
        return res;

    }

    private static void subsetsWithDupDfs(int[] nums, List<Integer> cur, int start, int end, List<List<Integer>> res, boolean[] used) {

        res.add(cur);

        if (start+1 == end){
            return;
        }



        for (int i = start; i < end; i++) {
            if (used[i]){
                continue;
            }
//            if (i>0 && end-start>1 && nums[i] == nums[i-1] && !used[i-1]){
//                continue;
//            }
            cur.add(nums[i]);
            used[i] = true;
            subsetsWithDupDfs(nums,cur,start+1,end,res,used);
            cur.remove(nums[i]);
            used[i] = false;
        }

    }


    public static void main(String[] args) {


//        System.out.println(maximalSquare(new char[][]{{'1'}}));
//        System.out.println(decodeString1("100[leetcode]"));
//        System.out.println(subarraySum(new int[]{-1,-1,1},0));
//        System.out.println(lengthOfLongestSubstring11("abcabcbb"));
//        System.out.println(longestPalindrome123("aaaa"));
//        System.out.println(exist(new char[][]{{'A','B','C','E'},{'S','F','E','S'},{'A','D','E','E'}},"ABCESEEEFS"));
//        System.out.println(trap1(new int[]{0,1,0,2,1,0,1,3,2,1,2,1}));
//        System.out.println(lengthOfLongestSubstring1234("pwwkew"));
//        System.out.println(rotate(new int[][]{{1,2,3},{4,5,6},{7,8,9}}));
//        System.out.println(myPow(2,10));
        System.out.println(subsetsWithDup(new int[]{1,2,2}));
//
//        ListNode listNode1 = new ListNode(2);
//        listNode1.setNext(new ListNode(4));
//        listNode1.getNext().setNext(new ListNode(3));
//
//
//
//        ListNode listNode2 = new ListNode(5);
//        listNode2.setNext(new ListNode(6));
//        listNode2.getNext().setNext(new ListNode(4));
//
//        System.out.println(addTwoNumbers(listNode1,listNode2));
//        System.out.println(twoSum(new int[]{3,2,4},6));
//        System.out.println(lengthOfLongestSubstring("pwwkew"));
//        System.out.println(lengthOfLongestSubstring1(" "));
//          nextPermutation(new int[]{2,3,1});
//        System.out.println(longestValidParentheses(")(())()()()())(((())))"));
//        System.out.println(combine(4,2));
    }
}

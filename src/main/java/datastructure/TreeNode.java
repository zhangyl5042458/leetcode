package datastructure;

import lombok.Data;
import org.apache.commons.collections4.CollectionUtils;

import java.lang.reflect.Array;
import java.util.*;

/**
 * @author: zhangyulin
 * @date: 2020-07-23 16:23
 * @description:
 */
@Data
public class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;
    TreeNode next;

    public TreeNode(int x) {
        val = x;
    }


    public static int maxPathSum(TreeNode root) {

        if (root == null){
            return 0;
        }
        int max = root.val;

        TreeNode left = root.left;
        TreeNode right = root.right;


        int leftsum = sum(0, root.left);
        int rightsum = sum(0, root.right);
        int val = root.val;
        max = Math.max(max,val+(leftsum>0?leftsum:0)+(rightsum>0?rightsum:0));
        while(left != null){
            int leftsum1 = sum(0, left.left);
            int rightsum1 = sum(0, left.right);
            max = Math.max(max,left.val+(leftsum1>0?leftsum1:0)+(rightsum1>0?rightsum1:0));
            left = left.left;
        }
        while(right != null){
            int leftsum1 = sum(0, right.left);
            int rightsum1 = sum(0, right.right);
            max = Math.max(max,right.val+(leftsum1>0?leftsum1:0)+(rightsum1>0?rightsum1:0));
            right = right.left;
        }
        return max;
    }


    public static int sum(int parentVal,TreeNode root) {
        if (root == null){
            return parentVal;
        }

        int leftVal = parentVal+root.val;
        int rightVal = parentVal+root.val;

        if (root.left != null){
            leftVal = sum(leftVal,root.left);
        }

        if (root.right != null){
            rightVal = sum(rightVal,root.right);
        }
        return Math.max(leftVal,rightVal);
    }


    public static List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();

        addRight(root,res);


        return res;
    }

    private static void addRight(TreeNode right, List<Integer> res) {


        if (right == null){
            return;
        }
        res.add(right.val);

        if (right.right!= null){
            addRight(right.right,res);
        }

    }


//    public static String decodeString(String s) {
//
//        StringBuilder sb = new StringBuilder();
//
//        Deque<Character> stack = new ArrayDeque<Character>();
//
//        for (int i = 0; i < s.length(); i++) {
//            stack.push(s.charAt(i));
//        }
//
//
//        while (!stack.isEmpty()){
//            Character poll = stack.poll();
//            if (poll.equals("[") || poll.equals("]")){
//
//            }
//            sb
//        }
//    }


    public static List<List<Integer>> levelOrder(TreeNode root) {

        List<List<Integer>> res = new ArrayList<>();
        if (root == null){
            return res;
        }



        Deque<Deque<TreeNode>> stack = new ArrayDeque<Deque<TreeNode>>();

        Deque<TreeNode> rootDeque = new ArrayDeque<TreeNode>();
        rootDeque.add(root);
        stack.add(rootDeque);


        while (!stack.isEmpty()){
            Deque<TreeNode> poll = stack.poll();
            List<Integer> lopRes = new ArrayList<>();

            Deque<TreeNode> addDeque = new ArrayDeque<>();
            while (!poll.isEmpty()){
                TreeNode poll1 = poll.poll();
                if (poll1.left!=null){
                    addDeque.add(poll1.left);
                }
                if (poll1.right!=null){
                    addDeque.add(poll1.right);
                }
                lopRes.add(poll1.val);
            }
            if (lopRes!=null && lopRes.size()>0){
                res.add(lopRes);
            }
            if (addDeque!=null && addDeque.size()>0){
                stack.add(addDeque);
            }

        }
        return res;
    }

    private static int maxVal = Integer.MIN_VALUE;


    public static int maxPathSum123(TreeNode root) {

        getChild(root);

        return maxVal;

    }

    private static int getChild(TreeNode root) {

        if (root.left == null && root.right == null){
            maxVal = Math.max(maxVal,root.val);
            return root.val;
        }


        int leftVal = 0;
        int rightVal = 0;
        if (root.left != null){
            leftVal= Math.max(getChild(root.left),0);
        }
        if (root.right != null){
            rightVal= Math.max(getChild(root.right),0);
        }

        int sum = root.val + leftVal + rightVal;

        maxVal = Math.max(sum,maxVal);

        return root.val+Math.max(leftVal,rightVal);

    }


    public static int divide(int dividend, int divisor) {
        boolean iszhengshu = (dividend>0&&divisor>0) || (dividend<0&&divisor<0);

        dividend = dividend>0?dividend:0-dividend;
        dividend = dividend==Integer.MIN_VALUE?Integer.MAX_VALUE:dividend;
        divisor = divisor>0?divisor:0-divisor;
        divisor = divisor==Integer.MIN_VALUE?Integer.MAX_VALUE:divisor;


        int sum = 0;

        int step = 0;

        while (true){
            sum += divisor;
            if (sum>dividend){
                break;
            }
            if (sum == dividend){
                ++step;
                break;
            }
            ++step;
        }

        return iszhengshu?step:0-step;

    }





    public static void main(String[] args) {
//        TreeNode root = new TreeNode(-10);
//        root.left = new TreeNode(9);
//        root.right = new TreeNode(-20);
//        root.right.left = new TreeNode(-15);
//        root.right.right = new TreeNode(7);
        System.out.println(divide(-2147483648,-1));
    }
}

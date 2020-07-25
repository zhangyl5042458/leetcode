package datastructure;

import lombok.Data;

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

    public TreeNode(int x) {
        val = x;
    }
}

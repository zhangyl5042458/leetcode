package datastructure;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Queue;

/**
 * @author: zhangyulin
 * @date: 2020-10-16 11:09
 * @description:
 */
public class Node {


    int val;
    Node left;
    Node right;
    Node next;

    public Node(int x) {
        val = x;
    }


    public Node connect(Node root) {

        if (root == null) {
            return root;
        }

        Node temp = root;

        while (temp.left != null && temp.right != null) {

            Node node = temp;

            while (node != null) {
                node.left.next = node.right;

                if (node.next != null) {
                    node.right.next = node.next.left;
                }

                node = node.next;
            }
            temp = temp.left;
        }

        return root;

    }


    public static Node connect123(Node root) {

        if (root == null) {
            return root;
        }

        Queue<Node> queue = new LinkedList<>();

        queue.add(root);

        while (!queue.isEmpty()) {

            int size = queue.size();


            for (int i = 0; i < size; i++) {

                Node next = queue.poll();

                if (i<size-1){
                    next.next = queue.peek();
                }

                if (next.left!= null){
                    queue.offer(next.left);
                }

                if (next.right!= null){
                    queue.offer(next.right);
                }
            }
        }
        return root;

    }


    public static void main(String[] args) {
        Node node = new Node(1);
        node.left = new Node(2);
        node.right = new Node(3);
        node.left.left = new Node(4);
        node.left.right = new Node(5);
        node.right.left = new Node(6);
        node.right.right = new Node(7);
        connect123(node);
    }

}

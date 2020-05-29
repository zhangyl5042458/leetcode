package topic;

import datastructure.ListNode;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-28 17:03
 * @Description:两数相加
 */
public class AddTwoNumbersSolution {
    /**
     *
     * @param l1
     * @param l2
     * @return
     */

    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        ListNode head = new ListNode(0);

        ListNode p = l1, q = l2, curr = head;

        //进位
        int carry = 0;

        while (p != null || q != null) {
            int x = p != null ? p.getVal() : 0;
            int y = q != null ? q.getVal() : 0;
            //当前位置sum
            int sum = x + y + carry;
            //进位
            carry = sum / 10;
            //设置下一位置的值
            curr.setNext(new ListNode(sum % 10));
            //当前位置下移
            curr = curr.getNext();

            if (p != null) {
                p = p.getNext();
            }

            if (q != null) {
                q = q.getNext();
            }
        }
        if (carry > 0) {
            //末位如果相加大于9
            curr.setNext(new ListNode(carry));
        }
        return head.getNext();
    }

    public static void main(String[] args) {
        System.out.println(addTwoNumbers(ListNode.getListNode(4, 5, 6), ListNode.getListNode(7, 8, 9)).toString());
        System.out.println(654 + 987);
    }
}

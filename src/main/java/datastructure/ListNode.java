package datastructure;

import lombok.Data;

import java.util.List;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-28 17:03
 * @Description:
 */
@Data
public class ListNode {

    public ListNode(Integer val){
        this.val = val;
    }


    public static ListNode getListNode(Integer... vals){
        ListNode curr = new ListNode(0);
        ListNode temp = curr;

        for (int i = 0; i < vals.length; i++) {
            temp.setVal(vals[i]);
            if (i != vals.length-1){
                temp.setNext(new ListNode(0));
                temp = temp.getNext();
            }
        }
        return curr;
    }

    private ListNode next;
    private Integer val;


    @Override
    public String toString() {
      return toString(this);
    }


    public String toString(ListNode listNode) {
        if (listNode == null){
            return null;
        }
        if (listNode.next!= null){
            return listNode.val.toString()+"->"+toString(listNode.next);
        }else{
            return listNode.val.toString();
        }
    }


    public ListNode removeNthFromEnd(ListNode head, int n) {

        ListNode slow = head;
        ListNode fast = head;

        ListNode res = slow;

        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        if (fast == null){
            return res.next;
        }

        while (fast.next!=null){
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return res;

    }

    public static void main(String[] args) {
        ListNode listNode = getListNode(1);
        System.out.println(listNode.toString());
    }
}

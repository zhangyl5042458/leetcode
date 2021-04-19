package datastructure;

import lombok.Data;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-28 17:03
 * @Description:
 */
@Data
public class ListNode {

    private ListNode listNode;

    public ListNode(Integer val) {
        this.val = val;
    }

    public static ListNode getListNode(Integer... vals) {
        ListNode curr = new ListNode(0);
        ListNode temp = curr;

        for (int i = 0; i < vals.length; i++) {
            temp.setVal(vals[i]);
            if (i != vals.length - 1) {
                temp.setNext(new ListNode(0));
                temp = temp.next;
            }
        }
        return curr;
    }


    public ListNode get(int index) {
        for (int i = 0; i <= index; i++) {
            listNode = listNode.next;
        }
        return listNode;
    }

    private ListNode pre;
    private ListNode next;
    private Integer val;


    @Override
    public String toString() {
        return toString(this);
    }


    public String toString(ListNode listNode) {
        if (listNode == null) {
            return null;
        }
        if (listNode.next != null) {
            return listNode.val.toString() + "->" + toString(listNode.next);
        } else {
            return listNode.val.toString();
        }
    }


    public static ListNode removeNthFromEnd1(ListNode head, int n) {

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


    public static class LRUCache {


        private Integer currentSize;
        private HashMap<Integer,ListNode> map;
        private Integer capacity;

        private ListNode head;
        private ListNode tail;


        public LRUCache(int capacity) {
            this.capacity  = capacity;
            map= new HashMap<>(capacity);
            currentSize = 0;
        }

        public int get(int key) {

            ListNode listNode = map.get(key);

            if (listNode == null){
                return -1;
            }
            listNode.pre.next = listNode.next;
            listNode.next.pre = listNode.pre;
            listNode.next = head;
            head.pre = listNode;
            listNode.pre = tail;
            tail.next = listNode;

            return listNode.val;
        }

        public void put(int key, int value) {

            ListNode newListNode = new ListNode(value);


            if (currentSize == 0){
                head = newListNode;
                tail = newListNode;
                head.next = tail;
                head.pre = tail;
                tail.next = head;
                tail.pre = head;
                map.put(key,newListNode);
                ++currentSize;
            }else{
                ListNode listNode = map.get(key);
                if (listNode != null){
                    listNode.pre.next = listNode.next;
                    listNode.next.pre = listNode.pre;
                    map.put(key,newListNode);
                }else{
                    if (currentSize == capacity){
                        map.remove(tail.val);
                        removeLast();
                    }
                    map.put(key,newListNode);
                    ++currentSize;
                }

                head.pre = newListNode;
                tail.next  = newListNode;

                newListNode.next = head;
                newListNode.pre = tail;
            }
        }

        private void removeLast() {
            tail.next.pre = tail.pre;
            tail.pre.next = tail.next;
            tail = tail.pre;
            --capacity;
        }
    }

    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        if (l1 == null) {
            return l2;
        }

        if (l2 == null) {
            return l1;
        }

        int jinwei = 0;

        int first = l1.val + l2.val;
        if (first >= 10) {
            jinwei = 1;
            first = first - 10;
        }

        ListNode dummyHead = new ListNode(0);

        ListNode firstNode = dummyHead;

        firstNode.next = new ListNode(first);

        firstNode = firstNode.next;

        l1 = l1.next;
        l2 = l2.next;

        while (l1 != null || l2 != null || jinwei!=0) {
            int l1Val = 0;
            int l2Val = 0;
            if (l1 != null) {
                l1Val = l1.val;
            }
            if (l2 != null) {
                l2Val = l2.val;
            }
            int sum = l1Val + l2Val + jinwei;
            if (sum >= 10) {
                jinwei = 1;
                sum = sum - 10;
            } else {
                jinwei = 0;
            }
            ListNode cur = new ListNode(sum);
            firstNode.next = cur;
            firstNode = firstNode.next;

            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }

        return dummyHead.next;
    }


    public static ListNode removeNthFromEnd(ListNode head, int n) {

        ListNode l1 = head;
        ListNode l2 = head;

        for (int i = 0; i < n; i++) {
            l1  = l1.next;
            if (l1 == null){
                //删除的是第一个元素
                return head.next;
            }
        }

        while (l1.next != null) {
            l1 = l1.next;
            l2 = l2.next;
        }

        l2.next = l2.next.next;

        return head;

    }


    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {

        ListNode listNode = new ListNode(0);

        ListNode curr = listNode;

        while (l1!= null && l2 != null){
            if (l1.val == l2.val){
                curr.next = new ListNode(l1.val);
                curr= curr.next;
                l1 = l1.next;
                curr.next = new ListNode(l2.val);
                curr= curr.next;
                l2 = l2.next;
            }else if (l1.val > l2.val){
                curr.next = new ListNode(l2.val);
                curr= curr.next;
                l2 = l2.next;
            }else if (l1.val < l2.val){
                curr.next = new ListNode(l1.val);
                curr= curr.next;
                l1 = l1.next;
            }
        }

        if (l1 == null){
            curr.next = l2;
        }


        if (l2 == null){
                curr.next = l1;
        }

        return listNode.next;

    }


    /**
     * 合并K个有序链表 hard
     * @param lists
     * @return
     */
    public static ListNode mergeKLists(ListNode[] lists) {

        ListNode result = null;

        for (int i = 0; i < lists.length; i++) {

            //两两合并
            result = mergeTwoLists(lists[i],result);
        }

        return result;

    }



    /**
     * 合并K个有序链表 hard 将全部元素放入集合中然后排序 在组成链表
     * @param lists
     * @return
     */
    public static ListNode mergeKLists1(ListNode[] lists) {

        ListNode result = new ListNode(0);

        ListNode cur = result;

        List<ListNode> list = new ArrayList<>();
        for (ListNode listNode : lists) {
            while (listNode != null) {
                list.add(new ListNode(listNode.val));
                listNode  = listNode.next;
            }
        }

        ListNode[] sortArr = quickSort(list.toArray(new ListNode[list.size()]));


        for (ListNode listNode : sortArr) {
            cur.next = listNode;
            cur = cur.next;
        }

        return result.next;

    }


    public static ListNode[] quickSort(ListNode[] arr){

        if (arr.length<=1){
            return arr;
        }

        quickSort(arr,0,arr.length-1);

        return arr;
    }


    public static ListNode[] quickSort(ListNode[] arr,int low,int high){

        int middle = getMiddle(arr, low, high);

        if (middle-1 >=low){
            quickSort(arr,low,middle-1);
        }

        if (middle+1 <=high){
            quickSort(arr,middle+1,high);
        }
        return arr;
    }



    private static int getMiddle(ListNode[] arr, int low, int high) {


        ListNode temp = arr[low];

        while (low<high){

            while (low<high && arr[high].val >= temp.val){
                --high;
            }

            ListNode tem = arr[high];
            arr[high] = arr[low];
            arr[low] = tem;


            while (low<high && arr[low].val <= temp.val){
                ++low;
            }

            ListNode tem1 = arr[low];
            arr[low] = arr[high];
            arr[high] = tem1;

        }

        return low;
    }


    /**
     *  链表反转 递归
     * @param listNode
     * @return
     */
    public static ListNode reverseList(ListNode listNode){


        if (listNode == null || listNode.next == null){
            return listNode;
        }

        ListNode next = reverseList(listNode.next);

        listNode.next.next = listNode;
        listNode.next = null;
        return next;


    }

    /**
     *  链表反转 双指针
     * @param listNode
     * @return
     */
    public static ListNode reverseList1(ListNode listNode){

        ListNode i = null,j = listNode;

        while (j!= null ){
            ListNode next = j.next;
            j.next = i;
            i=j;
            j = next;
        }

        return i;

    }


    /**
     *  链表反转 递归
     * @param listNode
     * @return
     */
    public static ListNode reverseList2(ListNode listNode){

        if (listNode == null || listNode.next == null){
            return listNode;
        }
        ListNode next = reverseList2(listNode.next);

        listNode.next.next = listNode;
        listNode.next = null;
        return next;
    }


    /**
     *  链表反转 双指针
     * @param listNode
     * @return
     */
    public static ListNode reverseList3(ListNode listNode){

        //初始化cur = null pre为头结点
        ListNode cur = null,pre = listNode;

        while (pre != null){
            //取出pre的next
            ListNode next = pre.next;
            //将pre的next纸箱前一个cur
            pre.next = cur;
            //分别将cur和pre向后移动一位  cur向后移动一位为pre ，pre向后移动一位为next 不能写成 pre=pre.next 因为此时pre.next = cur 链表城环了
            cur = pre;
            pre = next;
        }

        return cur;

    }

    /**
     *  环形链表  判断链表是否有环 并找到环的入口
     *
     *  https://mp.weixin.qq.com/s?__biz=MzUxNjY5NTYxNA==&mid=2247484171&idx=1&sn=72ba729f2f4b696dfc4987e232f1ad2d&scene=21#wechat_redirect
     *
     * @param listNode
     * @return
     */
    public static ListNode detectCycle(ListNode listNode){

        ListNode fast = listNode;
        ListNode slow = listNode;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow){
                ListNode index1 = listNode;
                ListNode index2 = fast;
                while (index1!= index2){
                    index1 = index1.next;
                    index2 = index2.next;
                }
                System.out.println(index2);
                return index2;
            }
        }
        return null;
    }




    public static ListNode removeElements (ListNode head,int val){

        ListNode first = new ListNode(0);

        first.next = head;

        while ( first.next!=null){
            if (first.next.val==val){
                first.next = first.next.next;
            }
            first = first.next;
        }
        return head;

    }


    /**
     * 链表反转 双指针
     * @return
     */
    public static ListNode reverseListDoubleIndex(ListNode head){

        ListNode i = null;
        ListNode j = head;


        while (j != null && j.next != null){
            ListNode next = j.next;
            j.next = i;
            i = j;
            j=next;
        }
        return i;
    }


    /**
     * 链表反转 递归
     * @return
     */
    public static ListNode reverseListRecursion(ListNode head){

        if (head == null || head.next == null){
            return head;
        }

        ListNode next = reverseListRecursion(head.next);

        head.next.next = head;
        head.next = null;

        return next;
    }


    public static ListNode detectCycle1 (ListNode listNode){

        ListNode slow = listNode;
        ListNode fast = listNode;

        while (fast != null && fast.next != null){

            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast){
                ListNode index1 = fast;
                ListNode index2 = listNode;
                while (index1 != index2){
                    index1 = index1.next;
                    index2 = index2.next;
                }
                return index2;
            }
        }
        return null;
    }





    public static ListNode reverseKGroup(ListNode head, int k) {

        ListNode cur = new ListNode(0);
        ListNode pre = head;

        ListNode d1 = cur;


        while (needReverse(pre,k)){
            //反转链表得到 新的反转数据和pre数据
           ListNode[] listData = reverseList(null,pre,k);
            d1.next =  listData[0];
            //后移k个
            for (int i = 0; i < k; i++) {
                d1 = d1.next;
            }
           pre = listData[1];
        }

        //剩下的拼起来
        d1.next = pre;
        return cur.next;


    }


    /**
     * 是否需要反转
     * @param listNode
     * @param k
     * @return
     */
    public static  boolean needReverse(ListNode listNode,int k){

        if (listNode == null){
            return false;
        }

        ListNode node = listNode;

        int len = 1;

        while(node.next!=null){
            len++;
            if (len>=k){
                return true;
            }
            node = node.next;
        }

        return false;

    }



    public static ListNode[] reverseList(ListNode cur,ListNode pre,int k){


        int length = 1;
        while (length<=k){
            ListNode next = pre.next;
            pre.next = cur;
            cur = pre;
            pre = next;
            length++;
        }
        return new ListNode[]{cur, pre};
    }



    public static ListNode addTwoNumbers123(ListNode l1, ListNode l2) {

        ListNode result = new ListNode(-1);

        ListNode temp = result;

        int jinwei = 0;

        while (l1 != null || l2 != null) {
            int l1val = l1 ==null?0:l1.val;
            int l2val = l2 ==null?0:l2.val;
            int i = l1val + l2val + jinwei;
            if (i >= 10) {
                jinwei = 1;
            } else {
                jinwei = 0;
            }
            temp.next = new ListNode(i % 10);
            l1 = l1==null?l1:l1.next;
            l2 = l2==null?l2:l2.next;
            temp = temp.next;
        }

        if (jinwei>0){
            temp.next = new ListNode(jinwei);
        }

        return result.next;

    }


    public ListNode mergeKLists123(ListNode[] lists) {

        return merge(lists,0,lists.length-1);
    }

    private ListNode merge(ListNode[] lists, int l, int r) {

        if (l == r){
            return lists[l];
        }

        if (l>r){
            return null;
        }

        int middle = (l+r)/2;
        return mergeTwoLists123(merge(lists,l,middle),merge(lists,middle+1,r));
    }

    public ListNode mergeTwoLists123(ListNode l1,ListNode l2) {

        ListNode result = new ListNode(-1);

        ListNode temp = result;

        while (l1!=null && l2 != null){
            if (l1.val < l2.val){
                result.next = new ListNode(l1.val);
                l1 = l1.next;
            }else{
                result.next = new ListNode(l2.val);
                l2 = l2.next;
            }
            result = result.next;
        }

        if (l1!=null){
            result.next = l1;
        }

        if (l2!=null){
            result.next = l2;
        }

        return temp.next;
    }



    public static ListNode sortList(ListNode head) {
        if (head == null || head.next == null){
            return head;
        }
        ListNode i = head;
        ListNode j = head.next;
        ListNode tempi = i;

        while (j!= null && j.next != null){
            i = i.next;
            j = j.next.next;
        }

        ListNode right = i.next;
        i.next = null;
        ListNode left = tempi;

        ListNode leftVal = sortList(left);
        ListNode rightVal = sortList(right);

        ListNode listNode = new ListNode(0);
        ListNode temp = listNode;

        while (leftVal!= null && rightVal!=null){
            if (leftVal.val < rightVal.val){
                listNode.next = new ListNode(leftVal.val);
                leftVal = leftVal.next;
            }else{
                listNode.next = new ListNode(rightVal.val);
                rightVal = rightVal.next;
            }
            listNode = listNode.next;
        }

        if(leftVal!=null){
            listNode.next = leftVal;
        }

        if(rightVal!=null){
            listNode.next = rightVal;
        }
        return temp.next;
    }


    public static ListNode removeNthFromEnd123(ListNode head, int n) {

        //计算链表总长度l 需要删除的元素为第 l-n+1 个元素

        ListNode loopNode = head;

        ListNode result = loopNode;

        int length = 0;

        while (head != null){
            ++length;
            head = head.next;
        }

        if (n == length){
            return loopNode.next;
        }

        int loopLength = 1;

        while (loopLength != length-n){
            loopNode = loopNode.next;
            ++loopLength;
        }

        loopNode.next = loopNode.next.next;

        return result;

    }


    public static ListNode removeNthFromEnd1234(ListNode head, int n) {

        //两个指针

        ListNode i1 = head;
        ListNode i2 = head;

        ListNode result = i2;

        for (int step = 0; step <= n; step++) {
            if (i1 == null){
                return result.next;
            }
            i1 = i1.next;
        }

        while (i1!=null){
            i1 = i1.next;
            i2 = i2.next;
        }

        i2.next = i2.next.next;

        return result;

    }


    /**
     * digui
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {

        if (head == null || head.next == null){
return head;
        }

        ListNode next = head.next;
        ListNode listNode = swapPairs(next.next);

        head.next  = listNode;
        next.next = head;

        return next;

    }

//    /**
//     * diedai
//     * @param head
//     * @return
//     */
//    public ListNode swapPairs1(ListNode head) {
//
//        ListNode temp = head;
//
//        while (temp.next != null && temp.next.next != null){
//
//            ListNode next = temp.next;
//            ListNode next1 = temp.next.next;
//        }
//    }


    public static void reorderList(ListNode head) {

        ListNode slow  =head;
        ListNode fast = head;
        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }


        ListNode temp = head;
        ListNode before = temp;

        while (temp.next!=null && temp.next != slow){
            temp = temp.next;
        }
        temp.next = null;



    }



    public static boolean isPalindrome(ListNode head) {

        ListNode fast = head;
        ListNode slow = head;

        while (fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }

        ListNode reverseListIsPalindromeNode = reverseListIsPalindrome(slow);

        while (reverseListIsPalindromeNode!=null){
            if (reverseListIsPalindromeNode.val != head.val){
                return false;
            }
            reverseListIsPalindromeNode = reverseListIsPalindromeNode.next;
            head = head.next;
        }
        return true;
    }

    private static ListNode reverseListIsPalindrome(ListNode slow) {

        ListNode prve = null;

        while (slow!= null){
            ListNode next = slow.next;
            slow.next = prve;
            prve = slow;
            slow = next;
        }

        return prve;


//        if (slow ==null || slow.next == null){
//            return slow;
//        }
//
//        ListNode next = slow.next;
//
//        ListNode listNode = reverseListIsPalindrome(next);
//
//        slow.next = null;
//        next.next = slow;
//
//        return listNode;


    }


    public static void main(String[] args) {
//        ListNode listNode = getListNode(1);
//        System.out.println(listNode.toString());

        ListNode listNode = new ListNode(4);
        listNode.next = new ListNode(2);
        listNode.next.next = new ListNode(1);
        listNode.next.next.next = new ListNode(2);
        listNode.next.next.next.next = new ListNode(4);

        System.out.println(isPalindrome(listNode));

        ListNode listNode1 = new ListNode(5);
        listNode1.next= new ListNode(6);
        listNode1.next.next = new ListNode(4);
        listNode1.next.next.next = new ListNode(9);
//        listNode.next.next.next.next.next = new ListNode(6);
//        listNode.next.next.next.next.next.next = listNode3;

//
//        ListNode node1 = new ListNode(1);
//        node1.next = new ListNode(4);
//        node1.next.next = new ListNode(5);
//        ListNode node2 = new ListNode(1);
//        node2.next = new ListNode(3);
//        node2.next.next = new ListNode(4);
//        ListNode node3 = new ListNode(2);
//        node3.next = new ListNode(6);
//        ListNode[] list = new ListNode[]{node1,node2,node3};
//
//        System.out.println(reverseKGroup(listNode,3));
        System.out.println(removeNthFromEnd1(listNode,1));

//        LRUCache lruCache = new LRUCache(2);
//        lruCache.put(1,1);
//        lruCache.put(2,2);
//        System.out.println(lruCache.get(1));
//        lruCache.put(3,3);
//        System.out.println(lruCache.get(2));
//        lruCache.put(4,4);
//        System.out.println(lruCache.get(1));
//        System.out.println(lruCache.get(3));
//        System.out.println(lruCache.get(4));


//
//        ListNode listNode2 = new ListNode(1);
//        listNode2.next = new ListNode(4);
//        listNode2.next.next = new ListNode(5);

//        System.out.println(detectCycle(listNode));
    }


}

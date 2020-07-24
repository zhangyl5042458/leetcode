package datastructure;

import java.util.Stack;

/**
 * @author: zhangyulin
 * @date: 2020-07-24 17:48
 * @description:
 */
public class CQueue {


    //stack1 插入元素  stack2 弹出元素
    private Stack<Integer> stack1 = new Stack<Integer>();
    private Stack<Integer> stack2 = new Stack<Integer>();

    public CQueue() {
    }

    public void appendTail(int value) {

        stack1.push(value);

    }

    public int deleteHead() {

        if (stack2.empty()){

            if (stack1.empty()){
                return -1;
            }else{
                while (!stack1.empty()){
                    stack2.push(stack1.pop());
                }
                return stack2.pop();
            }
        }else{

            return stack2.pop();
        }

    }
}

package topic;

import java.util.Arrays;
import java.util.HashMap;

/**
 * @author: zhangyulin
 * @date: 2020-09-15 15:06
 * @description:
 */
public class Search {


    public static int getIndexAndAdd(int[] arr , int tar,int low,int high){

        if (low == high){
            if (tar == arr[low]){
                return low;
            }else if(tar > arr[low]){
                return low+1;
            }else{
                return low-1 >=0?low-1:0;
            }
        }

        int middle = (high-low+1)/2;

        if (tar == arr[middle]){
            return middle;
        }else if (tar > arr[middle]){
            return getIndexAndAdd(arr, tar, middle, high);
        }else{
            return getIndexAndAdd(arr, tar, low, middle);
        }
    }


    /**
     * 移除元素
     *
     * 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
     *
     * 示例 1:
     * 给定 nums = [3,2,2,3], val = 3,
     * 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
     * 你不需要考虑数组中超出新长度后面的元素。
     *
     * 示例 2:
     * 给定 nums = [0,1,2,2,3,0,4,2], val = 2,
     * 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
     * @param a
     * @param tar
     * @return
     */
    public static int removeElement(int[] a,int tar){
        int len = 0;
        int i = 0;
        int j = 0;
        while (j<a.length){
            if (a[j] != tar){
               a[i] = a[j];
               ++i;
               ++j;
               len++;
            }else{
                if (j!= a.length-1){
                    a[i] = a[++j];
                }else {
                    a[i] = a[j];
                    ++j;
                }
            }
        }
        return len;
    }


    public static boolean threeSum(int[] a){

        Arrays.sort(a);

        for (int i = 0; i < a.length; i++) {
            if (a[i] >0){
                return false;
            }
            int tar = 0-a[i];

            if (i == a.length-1){
                return false;
            }

            int low = i+1;
            int high = a.length-1;
            while (low<high){
                if (a[low]+a[high] == tar){
                    return true;
                }else if (a[low]+a[high] > tar){
                    --high;
                }else{
                    ++low;
                }
            }
        }
        return false;
    }


    /**
     * 两数和 排序+双指针
     * @param a
     * @param tar
     * @return
     */
    public static boolean twoSum(int[] a,int tar){
        Arrays.sort(a);
        if (a.length<2){
            return false;
        }

        if (a.length<3){
            return a[0]+a[1]==tar;
        }
        int sum = a[0]+a[1];

        for (int i = 2; i < a.length; i++) {
            sum += a[i];
            sum -= a[i-2];
            if (sum == tar) {
                return true;
            }
        }

        return false;

    }


    /**
     * 两数和 哈希
     * @param a
     * @param tar
     * @return
     */
    public static boolean twoSumHash(int[] a,int tar){

        HashMap<Integer, Integer> map = new HashMap<>();


        for (int i = 0; i < a.length; i++) {
            map.put(a[i],i);
        }

        for (int i = 0; i < a.length; i++) {
            int diff = tar = a[i];
            if (map.containsKey(diff) && map.get(diff)!=i){
                return true;
            }
        }
        return false;

    }




    public static void main(String[] args) {
        System.out.println(removeElement(new int[]{0,1,2,3,3,0,4,2},2));
    }
}

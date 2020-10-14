package topic;

/**
 * @author: zhangyulin
 * @date: 2020-09-15 14:51
 * @description:
 */
public class Sort {


    public static int[] quickSort(int[] arr){

        if (arr.length<=1){
            return arr;
        }

        quickSort(arr,0,arr.length);

        return arr;
    }


    public static int[] quickSort(int[] arr,int low,int high){

        int middle = getMiddle(arr, low, high);
        quickSort(arr,low,middle-1);
        quickSort(arr,middle+1,high);

        return arr;
    }

    private static int getMiddle(int[] arr, int low, int high) {


        int temp = arr[low];

        while (low<high){

            while (low<high && arr[high] >= temp){
                --high;
            }

            int tem = arr[high];
            arr[high] = arr[low];
            arr[low] = tem;


            while (low<high && arr[low] <= temp){
                ++low;
            }

            tem = arr[low];
            arr[low] = arr[high];
            arr[high] = tem;

        }

        return low;
    }
}

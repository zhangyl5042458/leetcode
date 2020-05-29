package topic;

import java.util.HashMap;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 15:39
 * @Description:两数之和
 */
public class TwoSumSolution {

    public static int[] twoSum(int[] nums, int target) {

        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

        for (int i = 0; i < nums.length; i++) {

            int temp = target - nums[i];

            if (map.containsKey(temp)){
                return new int[] {map.get(temp),i};
            }
            map.put(nums[i],i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}

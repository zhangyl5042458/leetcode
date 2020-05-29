package topic;

import lombok.extern.slf4j.Slf4j;

import java.util.HashSet;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 10:30
 * @Description:无重复字符的最长子串
 */
public class LengthOfLongestSubstringSolution {

    public static int lengthOfLongestSubstring(String s) {

        HashSet<Character> characters = new HashSet<Character>();

        int n = s.length();

        int rk = -1 ,ans =0;

        for (int i = 0; i < n; i++) {

            if (i!=0){
                characters.remove(s.charAt(i-1));
            }

            while (rk+1 < n && !characters.contains(s.charAt(rk+1))){
                characters.add(s.charAt(rk+1));
                rk++;
            }

            ans = Math.max(ans,rk-i+1);

        }

        System.out.println(characters);

        return ans;
    }

    public static void main(String[] args) {
        System.out.println(lengthOfLongestSubstring("dsadsaqweqwe"));
    }
}

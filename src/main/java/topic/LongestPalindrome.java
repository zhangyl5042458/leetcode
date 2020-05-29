package topic;

/**
 * @Author: zhangyulin
 * @Date: 2020-05-29 17:09
 * @Description:最长回文子串
 */
public class LongestPalindrome {


    public static String longestPalindrome(String s) {
        if(s == null || s.trim().length() == 0){
            return "";
        }
        char[] c = s.toCharArray();
        int n = c.length;
        int t1,t2;
        int max = 0,start = 0,end = 0;
        for(int i = 0 ; i < n ; i++){
            t1 = i;
            t2 = i;
            while(t1 >= 0 && t2 < n && c[t1] == c[t2]){
                if(t2 - t1 > max){
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }

                t1--;
                t2++;
            }
            t1 = i;
            t2 = i + 1;
            while(t1 >= 0 && t2 < n && c[t1] == c[t2]){
                if(t2 - t1 > max){
                    max = t2 - t1;
                    start = t1;
                    end = t2;
                }
                t1--;
                t2++;
            }
        }
        return s.substring(start,end+1);
    }


    public static void main(String[] args) {
        System.out.println(longestPalindrome("qwedbaseddesbddd"));
    }
}

package datastructure;

import lombok.Data;

/**
 * @author: zhangyulin
 * @date: 2020-07-24 11:00
 * @description:
 */
@Data
public class TwoDimensionalArray {

    private int m;

    private int n;

    public TwoDimensionalArray(int m, int n) {
        this.m = m;
        this.n = n;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TwoDimensionalArray that = (TwoDimensionalArray) o;

        if (m != that.m) return false;
        return n == that.n;
    }

    @Override
    public int hashCode() {
        int result = m;
        result = 31 * result + n;
        return result;
    }
}

// time-limit: 2000
// problem-url: https://codeforces.com/contest/1919/problem/H
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
 
#define N   1000
#define W   1000000000
#define TEST    0
 
int min(int a, int b) { return a < b ? a : b; }
 
unsigned int X;
 
void srand_() {
    struct timeval tv;
 
    gettimeofday(&tv, NULL);
    X = tv.tv_sec ^ tv.tv_usec | 1;
    X = 1;
}
 
int rand_() {
    return (X *= 3) >> 1;
}
 
int n;
 
int *eh[N], eo[N], ii[N - 1], jj[N - 1], ww_[N - 1];
 
void append(int i, int h) {
    int o = eo[i]++;
 
    if (o >= 2 && (o & o - 1) == 0)
        eh[i] = (int *) realloc(eh[i], o * 2 * sizeof *eh[i]);
    eh[i][o] = h;
}
 
void load() {
    int h, i, j;
 
    for (i = 0; i < n; i++)
        eh[i] = (int *) malloc(2 * sizeof *eh[i]);
    for (h = 0; h < n - 1; h++) {
        scanf("%d%d", &i, &j), i--, j--;
        ii[h] = i, jj[h] = j;
        append(i, h), append(j, h);
    }
}
 
long long x_; int i_;
 
void dfs1(int p, int i, long long x) {
    int o;
 
    if (x_ < x)
        x_ = x, i_ = i;
    for (o = eo[i]; o--; ) { 
        int h = eh[i][o], j = i ^ ii[h] ^ jj[h], w = ww_[h];
 
        if (j != p)
            dfs1(i, j, x + w);
    }
}
 
int dfs2(int p, int i, int t) {
    int o;
 
    if (i == t)
        return 0;
    for (o = eo[i]; o--; ) {
        int h = eh[i][o], j = i ^ ii[h] ^ jj[h], d;
 
        if (j != p && (d = dfs2(i, j, t)) != -1)
            return d + 1;
    }
    return -1;
}
 
long long diameter(int *ww) {
    int h;
    long long x;
 
    printf("? 1");
    for (h = 0; h < n - 1; h++)
        printf(" %d", ww[h]);
    printf("\n"), fflush(stdout);
#if TEST
    memcpy(ww_, ww, (n - 1) * sizeof *ww);
    x_ = -1, i_ = -1, dfs1(-1, 0, 0);
    x_ = -1, dfs1(-1, i_, 0);
    x = x_;
    printf("> %lld\n", x);
#else
    scanf("%lld", &x);
#endif
    return x;
}
 
int distance(int h1, int h2) {
    int d;
 
    printf("? 2 %d %d\n", h1 + 1, h2 + 1), fflush(stdout);
#if TEST
    d = n;
    d = min(d, dfs2(-1, ii[h1], ii[h2]));
    d = min(d, dfs2(-1, ii[h1], jj[h2]));
    d = min(d, dfs2(-1, jj[h1], ii[h2]));
    d = min(d, dfs2(-1, jj[h1], jj[h2]));
    printf("> %d\n", d);
#else
    scanf("%d", &d);
#endif
    return d;
}
 
int dd[N - 1];
 
void sort(int *hh, int l, int r) {
    while (l < r) {
        int i = l, j = l, k = r, h = hh[l + rand_() % (r - l)], tmp;
 
        while (j < k)
            if (dd[hh[j]] == dd[h])
                j++;
            else if (dd[hh[j]] < dd[h]) {
                tmp = hh[i], hh[i] = hh[j], hh[j] = tmp;
                i++, j++;
            } else {
                k--;
                tmp = hh[j], hh[j] = hh[k], hh[k] = tmp;
            }
        sort(hh, l, i);
        l = k;
    }
}
 
int main() {
    static int hh[N - 1], ww[N - 1], ii[N - 1], jj[N - 1], hh0[N], hh0_[N], hh1[N], hh1_[N], aa[N], bb[N], pp[N];
    static char on[N];
    int m, cnt0, cnt0_, cnt1, cnt1_, h, hl, hr, h_, h__, h0, h0_, h1, h1_, idx, p, d, path, s;
    long long x;
 
    srand_();
    scanf("%d", &n);
#if TEST
    load();
#endif
    for (h = 1; h < n - 1; h++)
        dd[h] = distance(0, h);
    for (h = 1; h < n - 1; h++)
        hh[h] = h;
    sort(hh, 1, n - 1);
    m = 0;
    ii[m] = 0, jj[m] = 1, m++;
    cnt0 = cnt1 = 0, path = 1;
    for (d = 0, hl = 1; d < n && hl < n - 1; d++) {
        hr = hl;
        while (hr < n - 1 && dd[hh[hr]] == d)
            hr++;
        h__ = hh[hl];
        if (d == 0) {
            ii[m] = 0, jj[m] = h__ + 1, m++;
            pp[h__] = 0, hh0[cnt0++] = h__;
            for (h = hl + 1; h < hr; h++) {
                for (h_ = 0; h_ < n - 1; h_++)
                    ww[h_] = 1;
                h_ = hh[h];
                ww[0] = W, ww[h__] = W, ww[h_] = W;
                x = diameter(ww);
                if (x < (long long) W * 3) {
                    ii[m] = 0, jj[m] = h_ + 1, m++;
                    pp[h_] = 0, hh0[cnt0++] = h_;
                } else {
                    ii[m] = 1, jj[m] = h_ + 1, m++;
                    pp[h_] = 0, hh1[cnt1++] = h_;
                }
            }
            path = cnt0 == 1 && cnt1 == 1;
        } else {
            cnt0_ = cnt1_ = 0;
            if (cnt1 == 0) {
                for (h = hl; h < hr; h++) {
                    for (h_ = 0; h_ < n - 1; h_++)
                        ww[h_] = 1;
                    h_ = hh[h];
                    ww[0] = W, ww[h_] = W;
                    for (h0 = 0; h0 < cnt0; h0++) {
                        h0_ = hh0[h0];
                        ww[h0_] = (h0 + 1) * n;
                    }
                    x = diameter(ww) - W * 2;
                    h0 = x / n - 1, h0_ = hh0[h0];
                    ii[m] = h0_ + 1, jj[m] = h_ + 1, m++;
                    pp[h_] = h0_, hh0_[cnt0_++] = h_;
                }
            } else if (cnt0 == 0) {
                for (h = hl; h < hr; h++) {
                    for (h_ = 0; h_ < n - 1; h_++)
                        ww[h_] = 1;
                    h_ = hh[h];
                    ww[0] = W, ww[h_] = W;
                    for (h1 = 0; h1 < cnt1; h1++) {
                        h1_ = hh1[h1];
                        ww[h1_] = (h1 + 1) * n;
                    }
                    x = diameter(ww) - W * 2;
                    h1 = x / n - 1, h1_ = hh1[h1];
                    ii[m] = h1_ + 1, jj[m] = h_ + 1, m++;
                    pp[h_] = h1_, hh1_[cnt1_++] = h_;
                }
            } else {
                h__ = hh[hl];
                for (h = hl + 1; h < hr; h++) {
                    for (h_ = 0; h_ < n - 1; h_++)
                        ww[h_] = 1;
                    h_ = hh[h];
                    ww[h__] = W, ww[h_] = W;
                    for (h0 = 0; h0 < cnt0; h0++) {
                        h0_ = hh0[h0];
                        ww[h0_] = (h0 + 1) * n;
                    }
                    for (h1 = 0; h1 < cnt1; h1++) {
                        h1_ = hh1[h1];
                        ww[h1_] = (h1 + 1) * (cnt0 * 2 + 1) * n;
                    }
                    x = diameter(ww) - W * 2;
                    aa[h] = x / n % (cnt0 * 2 + 1), bb[h] = x / n / (cnt0 * 2 + 1);
                }
                if (path) {
                    ii[m] = hh0[0] + 1, jj[m] = h__ + 1, m++, hh0_[cnt0_++] = h__;
                    for (h = hl + 1; h < hr; h++) {
                        h_ = hh[h];
                        if (aa[h] == 0 && bb[h] == 0) {
                            ii[m] = hh0[0] + 1, jj[m] = h_ + 1, m++;
                            pp[h_] = hh0[0], hh0_[cnt0_++] = h_;
                        } else {
                            ii[m] = hh1[0] + 1, jj[m] = h_ + 1, m++;
                            pp[h_] = hh1[0], hh1_[cnt1_++] = h_;
                        }
                    }
                    path = cnt0_ == 1 && cnt1_ == 1;
                } else {
                    h = hl + 1;
                    while (h < hr && (aa[h] == 0 || bb[h] == 0))
                        h++;
                    s = -1, idx = -1, p = -1;
                    if (h == hr) {
                        if (distance(h__, hh[1]) <= d) {
                            s = 0;
                            for (h_ = 0; h_ < n - 1; h_++)
                                ww[h_] = 1;
                            ww[0] = W, ww[h__] = W;
                            for (h0 = 0; h0 < cnt0; h0++) {
                                h0_ = hh0[h0];
                                ww[h0_] = (h0 + 1) * n;
                            }
                            x = diameter(ww) - W * 2;
                            idx = x / n - 1, p = hh0[idx];
                            ii[m] = p + 1, jj[m] = h__ + 1, m++;
                            pp[h__] = p, hh0_[cnt0_++] = h__;
                        } else {
                            s = 1;
                            for (h_ = 0; h_ < n - 1; h_++)
                                ww[h_] = 1;
                            ww[0] = W, ww[h__] = W;
                            for (h1 = 0; h1 < cnt1; h1++) {
                                h1_ = hh1[h1];
                                ww[h1_] = (h1 + 1) * n;
                            }
                            x = diameter(ww) - W * 2;
                            idx = x / n - 1, p = hh1[idx];
                            ii[m] = p + 1, jj[m] = h__ + 1, m++;
                            pp[h__] = p, hh1_[cnt1_++] = h__;
                        }
                    } else {
                        memset(on, 0, (n - 1) * sizeof *on);
                        h0 = aa[h] - 1, h0_ = hh0[h0];
                        for (h_ = h0_; h_ != 0; h_ = pp[h_])
                            on[h_] = 1;
                        h1 = bb[h] - 1, h1_ = hh1[h1];
                        for (h_ = h1_; h_ != 0; h_ = pp[h_])
                            on[h_] = 1;
                        for (h = 1; h < n - 1; h++)
                            if (dd[h] < d && !on[h]) {
                                for (h_ = 0; h_ < n - 1; h_++)
                                    ww[h_] = 1;
                                ww[h__] = W, ww[h] = W;
                                ww[h0_] = (h0 + 1) * n;
                                ww[h1_] = (h1 + 1) * (cnt0 * 2 + 1) * n;
                                x = diameter(ww) - W * 2;
                                h0 = x / n % (cnt0 * 2 + 1) - 1, h1 = x / n / (cnt0 * 2 + 1) - 1;
                                if (h0 != -1) {
                                    s = 0;
                                    idx = h0, p = hh0[h0];
                                    ii[m] = p + 1, jj[m] = h__ + 1, m++;
                                    pp[h__] = p, hh0_[cnt0_++] = h__;
                                } else {
                                    s = 1;
                                    idx = h1, p = hh1[h1];
                                    ii[m] = p + 1, jj[m] = h__ + 1, m++;
                                    pp[h__] = p, hh1_[cnt1_++] = h__;
                                }
                                break;
                            }
                    }
                    for (h = hl + 1; h < hr; h++) {
                        h_ = hh[h];
                        if (aa[h] == 0 && bb[h] == 0) {
                            if (s == 0) {
                                ii[m] = p + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = p, hh0_[cnt0_++] = h_;
                            } else {
                                ii[m] = p + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = p, hh1_[cnt1_++] = h_;
                            }
                        } else if (aa[h] != 0 && bb[h] != 0) {
                            if (s == 1) {
                                h0 = aa[h] - 1, h0_ = hh0[h0];
                                ii[m] = h0_ + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = h0_, hh0_[cnt0_++] = h_;
                            } else {
                                h1 = bb[h] - 1, h1_ = hh1[h1];
                                ii[m] = h1_ + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = h1_, hh1_[cnt1_++] = h_;
                            }
                        } else {
                            if (s == 0) {
                                h0 = aa[h] - idx - 2, h0_ = hh0[h0];
                                ii[m] = h0_ + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = h0_, hh0_[cnt0_++] = h_;
                            } else {
                                h1 = bb[h] - idx - 2, h1_ = hh1[h1];
                                ii[m] = h1_ + 1, jj[m] = h_ + 1, m++;
                                pp[h_] = h1_, hh1_[cnt1_++] = h_;
                            }
                        }
                    }
                }
            }
            memcpy(hh0, hh0_, (cnt0 = cnt0_) * sizeof *hh0_);
            memcpy(hh1, hh1_, (cnt1 = cnt1_) * sizeof *hh1_);
        }   
        hl = hr;
    }
    printf("!\n");
    for (h = 0; h < n - 1; h++)
        printf("%d %d\n", ii[h] + 1, jj[h] + 1);
    fflush(stdout);
    return 0;
}









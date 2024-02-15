// time-limit: 5000
// problem-url: https://codeforces.com/contest/1919/problem/F2
// F2. Wine Factory (Hard Version) 

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>
using namespace std;

using ll = long long;
using db = long double;
using str = string;

// pairs
using pi = pair<int, int>;
using pl = pair<ll, ll>;
using pd = pair<db, db>;
#define mp make_pair
#define f first
#define s second

#define tcT template <class T
#define tcTU tcT, class U

tcT > using V = vector<T>;
tcT, size_t SZ > using AR = array<T, SZ>;
using vi = V<int>;
using vb = V<bool>;
using vl = V<ll>;
using vd = V<db>;
using vs = V<str>;
using vpi = V<pi>;
using vpl = V<pl>;
using vpd = V<pd>;

// vectors
#define sz(x) int((x).size())
#define bg(x) begin(x)
#define all(x) bg(x), end(x)
#define rall(x) x.rbegin(), x.rend()
#define sor(x) sort(all(x))
#define rsz resize
#define ins insert
#define pb push_back
#define eb emplace_back
#define ft front()
#define bk back()

#define lb lower_bound
#define ub upper_bound
tcT > int lwb(V<T> &a, const T &b) { return int(lb(all(a), b) - bg(a)); }
tcT > int upb(V<T> &a, const T &b) { return int(ub(all(a), b) - bg(a)); }

// loops
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define F0R(i, a) FOR(i, 0, a)
#define ROF(i, a, b) for (int i = (b)-1; i >= (a); --i)
#define R0F(i, a) ROF(i, 0, a)
#define rep(a) F0R(_, a)
#define each(a, x) for (auto &a : x)

const int MOD = 998244353;  // 1e9+7;
const int MX = (int)2e5 + 5;
const ll BIG = 1e18;  // not too close to LLONG_MAX
const db PI = acos((db)-1);
const int dx[4]{1, 0, -1, 0}, dy[4]{0, 1, 0, -1};  // for every grid problem!!
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
template <class T> using pqg = priority_queue<T, vector<T>, greater<T>>;

// bitwise ops
constexpr int pct(int x) { return __builtin_popcount(x); }  // # of bits set
constexpr int bits(int x) {

    return x == 0 ? 0 : 31 - __builtin_clz(x);
}
constexpr int p2(int x) { return 1 << x; }
constexpr int msk2(int x) { return p2(x) - 1; }

ll cdiv(ll a, ll b) {
    return a / b + ((a ^ b) > 0 && a % b);
}  // divide a by b rounded up
ll fdiv(ll a, ll b) {
    return a / b - ((a ^ b) < 0 && a % b);
}  // divide a by b rounded down

tcT > bool ckmin(T &a, const T &b) {
    return b < a ? a = b, 1 : 0;
}  // set a = min(a,b)
tcT > bool ckmax(T &a, const T &b) {
    return a < b ? a = b, 1 : 0;
}  // set a = max(a,b)

tcTU > T fstTrue(T lo, T hi, U f) {
    ++hi;
    assert(lo <= hi);  // assuming f is increasing
    while (lo < hi) {  // find first index such that f is true
        T mid = lo + (hi - lo) / 2;
        f(mid) ? hi = mid : lo = mid + 1;
    }
    return lo;
}
tcTU > T lstTrue(T lo, T hi, U f) {
    --lo;
    T d = 1;
    while (lo + d < hi) {
        if (!f(lo + d)) {
            hi = lo + d - 1;
            break;
        }
        d *= 2;
    }
    assert(lo <= hi);  // assuming f is decreasing
    while (lo < hi) {  // find first index such that f is true
        T mid = lo + (hi - lo + 1) / 2;
        f(mid) ? lo = mid : hi = mid - 1;
    }
    return lo;
}
tcT > void remDup(vector<T> &v) {  // sort and remove duplicates
    sort(all(v));
    v.erase(unique(all(v)), end(v));
}
tcTU > void safeErase(T &t, const U &u) {
    auto it = t.find(u);
    assert(it != end(t));
    t.erase(it);
}

inline namespace IO {
#define SFINAE(x, ...)                                                         \
    template <class, class = void> struct x : std::false_type {};              \
    template <class T> struct x<T, std::void_t<__VA_ARGS__>> : std::true_type {}

SFINAE(DefaultI, decltype(std::cin >> std::declval<T &>()));
SFINAE(DefaultO, decltype(std::cout << std::declval<T &>()));
SFINAE(IsTuple, typename std::tuple_size<T>::type);
SFINAE(Iterable, decltype(std::begin(std::declval<T>())));

template <auto &is> struct Reader {
    template <class T> void Impl(T &t) {
        if constexpr (DefaultI<T>::value) is >> t;
        else if constexpr (Iterable<T>::value) {
            for (auto &x : t) Impl(x);
        } else if constexpr (IsTuple<T>::value) {
            std::apply([this](auto &...args) { (Impl(args), ...); }, t);
        } else static_assert(IsTuple<T>::value, "No matching type for read");
    }
    template <class... Ts> void read(Ts &...ts) { ((Impl(ts)), ...); }
};

template <class... Ts> void re(Ts &...ts) { Reader<cin>{}.read(ts...); }
#define def(t, args...)                                                        \
    t args;                                                                    \
    re(args);

template <auto &os, bool debug, bool print_nd> struct Writer {
    string comma() const { return debug ? "," : ""; }
    template <class T> constexpr char Space(const T &) const {
        return print_nd && (Iterable<T>::value or IsTuple<T>::value) ? '\n'
                                                                     : ' ';
    }
    template <class T> void Impl(T const &t) const {
        if constexpr (DefaultO<T>::value) os << t;
        else if constexpr (Iterable<T>::value) {
            if (debug) os << '{';
            int i = 0;
            for (auto &&x : t)
                ((i++) ? (os << comma() << Space(x), Impl(x)) : Impl(x));
            if (debug) os << '}';
        } else if constexpr (IsTuple<T>::value) {
            if (debug) os << '(';
            std::apply(
                [this](auto const &...args) {
                    int i = 0;
                    (((i++) ? (os << comma() << " ", Impl(args)) : Impl(args)),
                     ...);
                },
                t);
            if (debug) os << ')';
        } else static_assert(IsTuple<T>::value, "No matching type for print");
    }
    template <class T> void ImplWrapper(T const &t) const {
        if (debug) os << "\033[0;31m";
        Impl(t);
        if (debug) os << "\033[0m";
    }
    template <class... Ts> void print(Ts const &...ts) const {
        ((Impl(ts)), ...);
    }
    template <class F, class... Ts>
    void print_with_sep(const std::string &sep, F const &f,
                        Ts const &...ts) const {
        ImplWrapper(f), ((os << sep, ImplWrapper(ts)), ...), os << '\n';
    }
    void print_with_sep(const std::string &) const { os << '\n'; }
};

template <class... Ts> void pr(Ts const &...ts) {
    Writer<cout, false, true>{}.print(ts...);
}
template <class... Ts> void ps(Ts const &...ts) {
    Writer<cout, false, true>{}.print_with_sep(" ", ts...);
}
}  // namespace IO

inline namespace Debug {
template <typename... Args> void err(Args... args) {
    Writer<cerr, true, false>{}.print_with_sep(" | ", args...);
}
template <typename... Args> void errn(Args... args) {
    Writer<cerr, true, true>{}.print_with_sep(" | ", args...);
}

void err_prefix(str func, int line, string args) {
    cerr << "\033[0;31m\u001b[1mDEBUG\033[0m"
         << " | "
         << "\u001b[34m" << func << "\033[0m"
         << ":"
         << "\u001b[34m" << line << "\033[0m"
         << " - "
         << "[" << args << "] = ";
}

#ifdef LOCAL
#define dbg(args...) err_prefix(__FUNCTION__, __LINE__, #args), err(args)
#define dbgn(args...) err_prefix(__FUNCTION__, __LINE__, #args), errn(args)
#else
#define dbg(...)
#define dbgn(args...)
#endif

const auto beg_time = std::chrono::high_resolution_clock::now();

double time_elapsed() {
    return chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    beg_time)
        .count();
}
}  // namespace Debug

inline namespace FileIO {
void setIn(str s) { freopen(s.c_str(), "r", stdin); }
void setOut(str s) { freopen(s.c_str(), "w", stdout); }
void setIO(str s = "") {
    cin.tie(0)->sync_with_stdio(0);  // unsync C / C++ I/O streams
    cout << fixed << setprecision(12);

    if (sz(s)) setIn(s + ".in"), setOut(s + ".out");
}
}  // namespace FileIO

/**
 * Description: Generate various types of trees.
 * Source: Own + Dhruv Rohatgi
 */
 
////////////// DISTRIBUTIONS
 
// return int in [L,R] inclusive
int rng_int(int L, int R) {
    assert(L <= R);
    return uniform_int_distribution<int>(L, R)(rng);
}
ll rng_ll(ll L, ll R) {
    assert(L <= R);
    return uniform_int_distribution<ll>(L, R)(rng);
}
 
// return double in [L,R] inclusive
db rng_db(db L, db R) {
    assert(L <= R);
    return uniform_real_distribution<db>(L, R)(rng);
}
 
// http://cplusplus.com/reference/random/geometric_distribution/geometric_distribution/
// flip a coin which is heads with probability p until you flip heads
// mean value of c is 1/p-1
int rng_geo(db p) {
    assert(0 < p && p <= 1);  // p large -> closer to 0
    return geometric_distribution<int>(p)(rng);
}
 
////////////// VECTORS + PERMS
 
// shuffle a vector
template <class T> void shuf(vector<T> &v) { shuffle(all(v), rng); }
 
// generate random permutation of [0,N-1]
vi randPerm(int N) {
    vi v(N);
    iota(all(v), 0);
    shuf(v);
    return v;
}
 
// random permutation of [0,N-1] with first element 0
vi randPermZero(int N) {
    vi v(N - 1);
    iota(all(v), 1);
    shuf(v);
    v.ins(bg(v), 0);
    return v;
}
 
// shuffle permutation of [0,N-1]
vi shufPerm(vi v) {
    int N = sz(v);
    vi key = randPerm(N);
    vi res(N);
    F0R(i, N) res[key[i]] = key[v[i]];
    return res;
}
 
// vector with all entries in [L,R]
vi rng_vec(int N, int L, int R) {
    vi res;
    F0R(_, N) res.pb(rng_int(L, R));
    return res;
}
 
// vector with all entries in [L,R], unique
vi rng_vec_unique(int N, int L, int R) {
    set<int> so_far;
    vi res;
    F0R(_, N) {
        int x;
        do { x = rng_int(L, R); } while (so_far.count(x));
        so_far.ins(x);
        res.pb(x);
    }
    return res;
}
 
////////////// GRAPHS
 
// relabel edges ed according to perm, shuffle
vpi relabelAndShuffle(vpi ed, vi perm) {
    each(t, ed) {
        t.f = perm[t.f], t.s = perm[t.s];
        if (rng() & 1) swap(t.f, t.s);
    }
    shuf(ed);
    return ed;
}
 
// shuffle graph with vertices [0,N-1]
vpi shufGraph(int N, vpi ed) {  // randomly swap endpoints, rearrange labels
    return relabelAndShuffle(ed, randPerm(N));
}
vpi shufGraphZero(int N, vpi ed) {
    return relabelAndShuffle(ed, randPermZero(N));
}
 
// shuffle tree given N-1 edges
vpi shufTree(vpi ed) { return shufGraph(sz(ed) + 1, ed); }
// randomly swap endpoints, rearrange labels
vpi shufRootedTree(vpi ed) {
    return relabelAndShuffle(ed, randPermZero(sz(ed) + 1));
}
 
void pgraphOne(int N, vpi ed) {
    ps(N, sz(ed));
    each(e, ed) ps(1 + e.f, 1 + e.s);
}
 
////////////// GENERATING TREES
 
// for generating tall tree
pi geoEdge(int i, db p) {
    assert(i > 0);
    return {i, max(0, i - 1 - rng_geo(p))};
}
 
// generate edges of tree with verts [0,N-1]
// smaller back -> taller tree
vpi treeRand(int N, int back) {
    assert(N >= 1 && back >= 0);
    vpi ed;
    FOR(i, 1, N) ed.eb(i, i - 1 - rng_int(0, min(back, i - 1)));
    return ed;
}
 
// generate path
vpi path(int N) { return treeRand(N, 0); }
 
// generate tall tree (large diameter)
// the higher the p the taller the tree
vpi treeTall(int N, db p) {
    assert(N >= 1);
    vpi ed;
    FOR(i, 1, N) ed.pb(geoEdge(i, p));
    return ed;
}
 
// generate tall tree, then add rand at end
vpi treeTallShort(int N, db p) {
    assert(N >= 1);
    int mid = (N + 1) / 2;
    vpi ed = treeTall(mid, p);
    FOR(i, mid, N) ed.eb(i, rng_int(0, i - 1));
    return ed;
}
 
// lots of stuff connected to either heavy1 or heavy2
vpi treeTallHeavy(int N, db p) {
    assert(N >= 1);  // + bunch of rand
    vpi ed;
    int heavy1 = 0, heavy2 = N / 2;
    FOR(i, 1, N) {
        if (i < N / 4) ed.eb(i, heavy1);
        else if (i > heavy2 && i < 3 * N / 4) ed.eb(i, heavy2);
        else ed.pb(geoEdge(i, p));
    }
    return ed;
}
 
// heavy tall tree + random
// lots of verts connected to heavy1 or heavy2
vpi treeTallHeavyShort(int N, db p) {
    assert(N >= 1);  // + almost-path + rand
    vpi ed;
    int heavy1 = 0, heavy2 = N / 2;
    FOR(i, 1, N) {
        if (i < N / 4) ed.eb(i, heavy1);
        else if (i <= heavy2) ed.pb(geoEdge(i, p)); 
        else if (i > heavy2 && i < 3 * N / 4) ed.eb(i, heavy2);
        else ed.eb(i, rng_int(0, i - 1));
    }
    return ed;
}
 
int rand_prime(int l, int r) {
    while (1) {
        int x = rng_int(l, r);
        bool bad = 0;
        for (int i = 2; i * i <= x; ++i)
            if (x % i == 0) bad = 1;
        if (!bad) return x;
    }
}
 
struct Info {
    ll v0 = 0, c0 = 0, c1 = BIG, ans = 0;
    pl eval(ll x) const {
        if (x <= c0) return {v0, ans + x};
        if (x <= c1) return {v0 + x - c0, ans + c0};
        return {v0 + c1 - c0, ans + c0};
    }
    pl eval(pl p) const {
        auto q = eval(p.f);
        return {q.f, q.s + p.s};
    }
};
 
Info rand_info() {
    ll c0 = rng_int(0, 10);
    ll c1 = rng_int(0, 10);
    if (c0 > c1) swap(c0, c1);
    return {rng_int(0, 10), c0, c1, rng_int(0, 10)};
}
 
ostream &operator<<(ostream &os, const Info &i) {
    os << "[" << i.v0 << " " << i.c0 << " " << i.c1 << " " << i.ans << "]";
    return os;
}
 
Info operator+(const Info &l, const Info &r) {
    auto [v0, ans] = r.eval({l.v0, l.ans});
    ll c0 = max(l.c0, r.c0 + l.c0 - l.v0);
    // l.v0 + x - l.c0 = r.c1
    ll c1 = min(l.c1, r.c1 + l.c0 - l.v0);
    Info i;
    if (c0 <= c1) {
        i = Info{v0, c0, c1, ans};
    } else {
        ll c = c0;
        ckmin(c, l.c1);
        i = Info{v0, c, c, ans};
    }
    return i;
}
 
/**
 * Description: 1D point update and range query where \texttt{cmb} is
 * any associative operation. \texttt{seg[1]==query(0,N-1)}.
 * Time: O(\log N)
 * Source:
 * http://codeforces.com/blog/entry/18051
 * KACTL
 * Verification: SPOJ Fenwick
 */
 
tcT > struct SegTree { 
    const T ID{};
    T cmb(T a, T b) { return a + b; }
    int n;
    V<T> seg;
    void init(int _n) { 
        for (n = 1; n < _n;) n *= 2;
        seg.assign(2 * n, ID);
    }
    void pull(int p) { seg[p] = cmb(seg[2 * p], seg[2 * p + 1]); }
    void upd(int p, T val) {  // set val at position p
        seg[p += n] = val;
        for (p /= 2; p; p /= 2) pull(p);
    }
    T query(int l, int r) {
        T ra = ID, rb = ID;
        for (l += n, r += n + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) ra = cmb(ra, seg[l++]);
            if (r & 1) rb = cmb(seg[--r], rb);
        }
        return cmb(ra, rb);
    }
};
 
vi A;
vi B;
vl C;
SegTree<Info> seg;
 
int main() {
    setIO();
    def(int, N, Q);
    A.rsz(N), B.rsz(N), C.rsz(N);
    re(A, B);
    F0R(i, N - 1) re(C[i]);
    seg.init(N);
    auto update = [&](int i) {
        seg.upd(i, Info{A[i], 0, BIG, 0} + Info{0, B[i], BIG, 0} +
                       Info{0, 0, C[i], 0});
    };
    F0R(i, N) update(i);
    dbg(seg.seg.at(1));
    rep(Q) {
        def(int, p, x, y);
        def(ll, z);
        --p;
        A.at(p) = x;
        B.at(p) = y;
        C.at(p) = z;
        update(p);
        ps(seg.seg.at(1).ans);
    } 
}










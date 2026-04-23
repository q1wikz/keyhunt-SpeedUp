/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/BSGS).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Int.h"
#include <emmintrin.h>
#include <string.h>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

static Int     _P;       // Field characteristic
static Int     _R;       // Montgomery multiplication R
static Int     _R2;      // Montgomery multiplication R2
static Int     _R3;      // Montgomery multiplication R3
static Int     _R4;      // Montgomery multiplication R4
static int32_t  Msize;    // Montgomery mult size
static uint32_t MM32;     // 32bits lsb negative inverse of P
static uint64_t MM64;     // 64bits lsb negative inverse of P
#define MSK62  0x3FFFFFFFFFFFFFFF

extern Int _ONE;

// ------------------------------------------------

void Int::ModAdd(Int *a) {
  Int p;
  Add(a);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModAdd(Int *a, Int *b) {
  Int p;
  Add(a,b);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModDouble() {
  Int p;
  Add(this);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModAdd(uint64_t a) {
  Int p;
  Add(a);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModSub(Int *a) {
  Sub(a);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModSub(uint64_t a) {
  Sub(a);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModSub(Int *a,Int *b) {
  Sub(a,b);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModNeg() {
  Neg();
  Add(&_P);
}

// ------------------------------------------------

void Int::ModInv() {

  // Compute modular inverse of this mop _P
  // 0 < this < P  , P must be odd
  // Return 0 if no inverse

  // 256bit 
  //#define XCD 1               // ~62  kOps/s
  //#define BXCD 1              // ~167 kOps/s
  //#define MONTGOMERY 1        // ~200 kOps/s
  //#define PENK 1              // ~179 kOps/s
  #define DRS62 1             // ~365 kOps/s

  Int u(&_P);
  Int v(this);
  Int r((int64_t)0);
  Int s((int64_t)1);

#ifdef XCD

  Int q, t1, t2, w;

  // Classic XCD 

  bool bIterations = true;  // Remember odd/even iterations
  while (!u.IsZero()) {
    // Step X3. Divide and "Subtract"
    q.Set(&v);
    q.Div(&u, &t2);   // q = u / v, t2 = u % v
    w.Mult(&q, &r);   // w = q * r
    t1.Add(&s, &w);   // t1 = s + w

                      // Swap u,v & r,s
    s.Set(&r);
    r.Set(&t1);
    v.Set(&u);
    u.Set(&t2);

    bIterations = !bIterations;
  }

  if (!v.IsOne()) {
    CLEAR();
    return;
  }

  if (!bIterations) {
    Set(&_P);
    Sub(&s);  /* inv = n - u1 */
  } else {
    Set(&s);  /* inv = u1     */
  }

#endif

#ifdef BXCD

#define SWAP_SUB(x,y) x.Sub(&y);y.Add(&x)

  // Binary XCD loop

  while (!u.IsZero()) {

    if (u.IsEven()) {

      u.ShiftR(1);
      if (r.IsOdd())
        r.Add(&_P);
      r.ShiftR(1);

    } else {

      SWAP_SUB(u, v);
      SWAP_SUB(r, s);

    }

  }

  // v ends with -1 ou 1
  if (!v.IsOne()) {
    // v = -1
    s.Neg();
    s.Add(&_P);
    v.Neg();
  }

  if (!v.IsOne()) {
    CLEAR();
    return;
  }

  if (s.IsNegative())
    s.Add(&_P);

  if (s.IsGreaterOrEqual(&_P))
    s.Sub(&_P);

  Set(&s);

#endif

#ifdef PENK

  Int x;
  Int n2(&_P);
  int k = 0;
  int T;
  int Q = _P.bits[0] & 3;
  shiftL(1,n2.bits64);

  // Penk's Algorithm (With DRS2 optimisation)

  while (v.IsEven()) {

    shiftR(1,v.bits64);
    if (s.IsEven())
      shiftR(1, s.bits64);
    else if (s.IsGreater(&_P)) {
      s.Sub(&_P);
      shiftR(1, s.bits64);
    } else {
      s.Add(&_P);
      shiftR(1, s.bits64);
    }

  }

  while (true) {

    if (u.IsGreater(&v)) {

      if ((u.bits[0] & 2) == (v.bits[0] & 2)) {
        u.Sub(&v);
        r.Sub(&s);
      } else {
        u.Add(&v);
        r.Add(&s);
      }
      shiftR(2,u.bits64);
      T = r.bits[0] & 3;
      if (T == 0) {
        shiftR(2,r.bits64);
      } else if (T == 2) {
        r.Add(&n2);
        shiftR(2, r.bits64);
      } else if (T == Q) {
        r.Sub(&_P);
        shiftR(2, r.bits64);
      } else {
        r.Add(&_P);
        shiftR(2, r.bits64);
      }
      while (u.IsEven()) {
        shiftR(1,u.bits64);
        if (r.IsEven()) {
          shiftR(1, r.bits64);
        } else if (r.IsGreater(&_P)) {
          r.Sub(&_P);
          shiftR(1, r.bits64);
        } else {
          r.Add(&_P);
          shiftR(1, r.bits64);
        }
      }

    } else {

      if ((u.bits[0] & 2) == (v.bits[0] & 2)) {
        v.Sub(&u);
        s.Sub(&r);
      } else {
        v.Add(&u);
        s.Add(&r);
      }

      if (v.IsZero())
        break;

      shiftR(2, v.bits64);
      T = s.bits[0] & 3;
      if (T == 0) {
        shiftR(2,s.bits64);
      } else if (T == 2) {
        s.Add(&n2);
        shiftR(2, s.bits64);
      } else if (T == Q) {
        s.Sub(&_P);
        shiftR(2, s.bits64);
      } else {
        s.Add(&_P);
        shiftR(2, s.bits64);
      }

      while (v.IsEven()) {
        shiftR(1, v.bits64);
        if (s.IsEven()) {
          shiftR(1, s.bits64);
        } else if (s.IsGreater(&_P)) {
          s.Sub(&_P);
          shiftR(1, s.bits64);
        } else {
          s.Add(&_P);
          shiftR(1, s.bits64);
        }
      }

    }

  }

  if (u.IsGreater(&_ONE)) {
    CLEAR();
    return;
  }
  if (r.IsNegative())
    r.Add(&_P);
  Set(&r);

#endif

#ifdef MONTGOMERY

  Int x;
  int k = 0;

  // Montgomery method
  while (v.IsStrictPositive()) {
    if (u.IsEven()) {
      shiftR(1, u.bits64);
      shiftL(1, s.bits64);
    } else if (v.IsEven()) {
      shiftR(1, v.bits64);
      shiftL(1, r.bits64);
    } else {
      x.Set(&u);
      x.Sub(&v);
      if (x.IsStrictPositive()) {
        shiftR(1, x.bits64);
        u.Set(&x);
        r.Add(&s);
        shiftL(1, s.bits64);
      } else {
        x.Neg();
        shiftR(1, x.bits64);
        v.Set(&x);
        s.Add(&r);
        shiftL(1, r.bits64);
      }
    }
    k++;
  }

  if (r.IsGreater(&_P))
    r.Sub(&_P);
  r.Neg();
  r.Add(&_P);

  for (int i = 0; i < k; i++) {
    if (r.IsEven()) {
      shiftR(1, r.bits64);
    } else {
      r.Add(&_P);
      shiftR(1, r.bits64);
    }
  }
  Set(&r);

#endif

#ifdef DRS62

  // Delayed right shift 62bits

  #define SWAP_ADD(x,y) x+=y;y-=x;
  #define SWAP_SUB(x,y) x-=y;y+=x;
  #define IS_EVEN(x) ((x&1)==0)

  Int r0_P;
  Int s0_P;
  Int uu_u;
  Int uv_v;
  Int vu_u;
  Int vv_v;
  Int uu_r;
  Int uv_s;
  Int vu_r;
  Int vv_s;

  int64_t bitCount;
  int64_t uu, uv, vu, vv;
  int64_t v0, u0;
  int64_t nb0;

  while (!u.IsZero()) {

    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of 
    // 'added P' can be easily calculated
    uu = 1; uv = 0;
    vu = 0; vv = 1;

    u0 = (int64_t)u.bits64[0];
    v0 = (int64_t)v.bits64[0];
    bitCount = 0;

    // Slightly optimized Binary XCD loop on native signed integers
    // Stop at 62 bits to avoid uv matrix overfow and loss of sign bit
    while (true) {

      while (IS_EVEN(u0) && bitCount<62) {

        bitCount++;
        u0 >>= 1;
        vu <<= 1;
        vv <<= 1;

      }

      if (bitCount == 62)
        break;

      nb0 = (v0 + u0) & 0x3;
      if (nb0 == 0) {
        SWAP_ADD(uv, vv);
        SWAP_ADD(uu, vu);
        SWAP_ADD(u0, v0);
      } else {
        SWAP_SUB(uv, vv);
        SWAP_SUB(uu, vu);
        SWAP_SUB(u0, v0);
      }

    }

    // Now update BigInt variables

    uu_u.IMult(&u,uu);
    uv_v.IMult(&v,uv);

    vu_u.IMult(&u,vu);
    vv_v.IMult(&v,vv);

    uu_r.IMult(&r,uu);
    uv_s.IMult(&s,uv);

    vu_r.IMult(&r,vu);
    vv_s.IMult(&s,vv);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    uint64_t r0 = ((uu_r.bits64[0] + uv_s.bits64[0]) * MM64) & MSK62;
    uint64_t s0 = ((vu_r.bits64[0] + vv_s.bits64[0]) * MM64) & MSK62;
    r0_P.Mult(&_P,r0);
    s0_P.Mult(&_P,s0);

    // u = (uu*u + uv*v)
    u.Add(&uu_u,&uv_v);

    // v = (vu*u + vv*v)
    v.Add(&vu_u,&vv_v);

    // r = (uu*r + uv*s + r0*P)
    r.Add(&uu_r,&uv_s);
    r.Add(&r0_P);

    // s = (vu*r + vv*s + s0*P)
    s.Add(&vu_r,&vv_s);
    s.Add(&s0_P);

    // Right shift all variables by 62bits
    shiftR(62, u.bits64);
	shiftR(62, v.bits64);
	shiftR(62, r.bits64);
	shiftR(62, s.bits64);

  }

  // v ends with -1 or 1
  if (v.IsNegative()) {
    // V = -1
    v.Neg();
    s.Neg();
    s.Add(&_P);
  }
  if (!v.IsOne()) {
    // No inverse
    CLEAR();
    return;
  }

  if (s.IsNegative())
    s.Add(&_P);

  if (s.IsGreaterOrEqual(&_P))
    s.Sub(&_P);

  Set(&s);

#endif

}

// ------------------------------------------------

void Int::ModExp(Int *e) {
  Int base(this);
  SetInt32(1);
  uint32_t nbBit = e->GetBitLength();
  for(int i=0;i<(int)nbBit;i++) {
    if (e->GetBit(i))
      ModMul(&base);
    base.ModMul(&base);
  }

}

// ------------------------------------------------

void Int::ModMul(Int *a) {

  Int p;
  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);

}

// ------------------------------------------------

void Int::ModSquare(Int *a) {

  Int p;
  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);

}

// ------------------------------------------------

void Int::ModCube(Int *a) {

  Int p;
  Int p2;
  p.MontgomeryMult(a, a);
  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);

}

// ------------------------------------------------

bool Int::HasSqrt() {

  // Euler's criterion
  Int e(&_P);
  Int a(this);
  e.SubOne();
  e.ShiftR(1);
  a.ModExp(&e);

  return a.IsOne();

}

// ------------------------------------------------

void Int::ModSqrt() {

  if (_P.IsEven()) {
    CLEAR();
    return;
  }

  if (!HasSqrt()) {
    CLEAR();
    return;
  }

  if ((_P.bits64[0] & 3) == 3) {

    Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);

  } else if ((_P.bits64[0] & 3) == 1) {

    //int nbBit = _P.GetBitLength();

    // Tonelli Shanks
    uint64_t e=0;
    Int S(&_P);
    S.SubOne();
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    // Search smalest non-qresidue of P
    Int q((uint64_t)1);
    do {
      q.AddOne();
    }  while (q.HasSqrt());

    Int c(&q);
    c.ModExp(&S);

    Int t(this);
    t.ModExp(&S);

    Int r(this);
    Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;
    while (!t.IsOne()) {

      Int t2(&t);
      uint64_t i=0;
      while (!t2.IsOne()) {
        t2.ModSquare(&t2);
        i++;
      }

      Int b(&c);
      for(uint64_t j=0;j<M-i-1;j++)
        b.ModSquare(&b);
      M=i;
      c.ModSquare(&b);
      t.ModMul(&t,&c);
      r.ModMul(&r,&b);

    }

    Set(&r);

  }

}

// ------------------------------------------------

void Int::ModMul(Int *a, Int *b) {

  Int p;
  p.MontgomeryMult(a,b);
  MontgomeryMult(&_R2,&p);

}

// ------------------------------------------------

Int* Int::GetFieldCharacteristic() {
  return &_P;
}

// ------------------------------------------------

Int* Int::GetR() {
  return &_R;
}
Int* Int::GetR2() {
  return &_R2;
}
Int* Int::GetR3() {
  return &_R3;
}
Int* Int::GetR4() {
  return &_R4;
}

// ------------------------------------------------

void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {

  // Size in number of 32bit word
  int nSize = n->GetSize();

  // Last digit inversions (Newton's iteration)
  {
    int64_t x, t;
    x = t = (int64_t)n->bits64[0];
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    MM64 = (uint64_t)(-x);
    MM32 = (uint32_t)MM64;
  }
  _P.Set(n);

  // Size of Montgomery mult (64bits digit)
  Msize = nSize/2;

  // Compute few power of R
  // R = 2^(64*Msize) mod n
  Int Ri;
  Ri.MontgomeryMult(&_ONE, &_ONE); // Ri = R^-1
  _R.Set(&Ri);                     // R  = R^-1
  _R2.MontgomeryMult(&Ri, &_ONE);  // R2 = R^-2
  _R3.MontgomeryMult(&Ri, &Ri);    // R3 = R^-3
  _R4.MontgomeryMult(&_R3, &_ONE); // R4 = R^-4

  _R.ModInv();                     // R  = R
  _R2.ModInv();                    // R2 = R^2
  _R3.ModInv();                    // R3 = R^3
  _R4.ModInv();                    // R4 = R^4

  if (R)
    R->Set(&_R);

  if (R2)
    R2->Set(&_R2);

  if (R3)
    R3->Set(&_R3);

  if (R4)
    R4->Set(&_R4);

}
// ------------------------------------------------

uint64_t Int::AddC(Int *a) {

  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif

  return c;

}

// ------------------------------------------------

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {

  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 + 0);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 + 1);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 + 2);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 + 4);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 + 5);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 + 6);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif

  bits64[NB64BLOCK-1] = c + cH;

}

// ------------------------------------------------
void Int::MontgomeryMult(Int *a) {

  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < Msize; i++) {

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
	  c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t,&_P);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

}

void Int::MontgomeryMult(Int *a, Int *b) {

  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, b->bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(bits64,pr.bits64 + 1,8*(NB64BLOCK-1));
  bits64[NB64BLOCK-1] = c;

  for (int i = 1; i < Msize; i++) {

    imm_umul(a->bits64, b->bits64[i], pr.bits64);
    ML = (pr.bits64[0] + bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
	  c = pr.AddC(&p);
    AddAndShift(this, &pr, c);

  }

  p.Sub(this, &_P);
  if (p.IsPositive())
    Set(&p);

}


// SecpK1 specific section -----------------------------------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {

#if defined(__GNUC__) && defined(__BMI2__) && defined(__ADX__)

  // ADX+BMI2 optimized 256x256 mod P multiply for secp256k1
  // Uses mulx (BMI2) + adcx/adox (ADX) for dual carry chains
  uint64_t *aa = a->bits64;
  uint64_t *bb = b->bits64;
  uint64_t *rr = bits64;

  __asm__ __volatile__ (
    // r512[0..7] will be built in: r8,r9,r10,r11,r12,r13,r14,r15
    // rdx = multiplier for mulx

    // ---- Column 0: b[0] * a[0..3] ----
    "movq (%[B]), %%rdx\n\t"        // rdx = b[0]
    "mulx (%[A]), %%r8, %%r9\n\t"   // r8=lo, r9=hi  (r512[0], carry)
    "mulx 8(%[A]), %%rax, %%r10\n\t" // rax=lo, r10=hi
    "addq %%rax, %%r9\n\t"
    "mulx 16(%[A]), %%rax, %%r11\n\t"
    "adcq %%rax, %%r10\n\t"
    "mulx 24(%[A]), %%rax, %%r12\n\t"
    "adcq %%rax, %%r11\n\t"
    "adcq $0, %%r12\n\t"

    // ---- Column 1: b[1] * a[0..3] ----
    "movq 8(%[B]), %%rdx\n\t"       // rdx = b[1]
    "xorl %%r15d, %%r15d\n\t"       // r15 = 0, also clears OF for adox
    "mulx (%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r9\n\t"         // r9 += lo (CF chain)
    "adoxq %%rcx, %%r10\n\t"        // r10 += hi (OF chain)
    "mulx 8(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r10\n\t"
    "adoxq %%rcx, %%r11\n\t"
    "mulx 16(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r11\n\t"
    "adoxq %%rcx, %%r12\n\t"
    "mulx 24(%[A]), %%rax, %%r13\n\t"
    "adcxq %%rax, %%r12\n\t"
    "adoxq %%r15, %%r13\n\t"        // r13 += 0 + OF
    "adcxq %%r15, %%r13\n\t"        // r13 += CF

    // ---- Column 2: b[2] * a[0..3] ----
    "movq 16(%[B]), %%rdx\n\t"
    "xorl %%r15d, %%r15d\n\t"
    "mulx (%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r10\n\t"
    "adoxq %%rcx, %%r11\n\t"
    "mulx 8(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r11\n\t"
    "adoxq %%rcx, %%r12\n\t"
    "mulx 16(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r12\n\t"
    "adoxq %%rcx, %%r13\n\t"
    "mulx 24(%[A]), %%rax, %%r14\n\t"
    "adcxq %%rax, %%r13\n\t"
    "adoxq %%r15, %%r14\n\t"
    "adcxq %%r15, %%r14\n\t"

    // ---- Column 3: b[3] * a[0..3] ----
    "movq 24(%[B]), %%rdx\n\t"
    "xorl %%r15d, %%r15d\n\t"
    "mulx (%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r11\n\t"
    "adoxq %%rcx, %%r12\n\t"
    "mulx 8(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r12\n\t"
    "adoxq %%rcx, %%r13\n\t"
    "mulx 16(%[A]), %%rax, %%rcx\n\t"
    "adcxq %%rax, %%r13\n\t"
    "adoxq %%rcx, %%r14\n\t"
    "mulx 24(%[A]), %%rax, %%r15\n\t"
    "adcxq %%rax, %%r14\n\t"
    "movq $0, %%rax\n\t"
    "adoxq %%rax, %%r15\n\t"
    "adcxq %%rax, %%r15\n\t"

    // Now r512 = r8:r9:r10:r11 (low) : r12:r13:r14:r15 (high)

    // ---- Reduce 512->320: high * 0x1000003D1 + low ----
    "movq $0x1000003D1, %%rdx\n\t"

    "mulx %%r12, %%rax, %%rcx\n\t"  // r12 * K
    "addq %%rax, %%r8\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r12\n\t"         // save carry

    "mulx %%r13, %%rax, %%rcx\n\t"  // r13 * K
    "addq %%r12, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r9\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r12\n\t"

    "mulx %%r14, %%rax, %%rcx\n\t"  // r14 * K
    "addq %%r12, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r10\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r12\n\t"

    "mulx %%r15, %%rax, %%rcx\n\t"  // r15 * K
    "addq %%r12, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r11\n\t"
    "adcq $0, %%rcx\n\t"

    // ---- Reduce 320->256: rcx * 0x1000003D1 ----
    "mulx %%rcx, %%rax, %%rcx\n\t"
    "addq %%rax, %%r8\n\t"
    "adcq %%rcx, %%r9\n\t"
    "adcq $0, %%r10\n\t"
    "adcq $0, %%r11\n\t"

    // Store result
    "movq %%r8, (%[R])\n\t"
    "movq %%r9, 8(%[R])\n\t"
    "movq %%r10, 16(%[R])\n\t"
    "movq %%r11, 24(%[R])\n\t"
    "movq $0, 32(%[R])\n\t"

    : /* no output operands */
    : [A] "r" (aa), [B] "r" (bb), [R] "r" (rr)
    : "rax", "rcx", "rdx",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "cc", "memory"
  );

#else
  // Fallback: original intrinsics implementation
  unsigned char c;

  uint64_t ah, al;
  uint64_t t[5];
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  imm_umul(a->bits64, b->bits64[0], r512);
  imm_umul(a->bits64, b->bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);
  imm_umul(a->bits64, b->bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);
  imm_umul(a->bits64, b->bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320 
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah); 
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0; 
#endif

}

void Int::ModMulK1(Int *a) {

  // this = this * a (mod P), just delegates to 3-arg version
  ModMulK1(a, this);

}

void Int::ModSquareK1(Int *a) {

#if defined(__GNUC__) && defined(__BMI2__) && defined(__ADX__)

  // ADX+BMI2 optimized 256-bit squaring mod P for secp256k1
  // Squaring: compute off-diagonal products, double them, add diagonal squares
  // a[0..3] squared -> r512[0..7], then reduce mod P
  uint64_t *aa = a->bits64;
  uint64_t *rr = bits64;

  __asm__ __volatile__ (
    // Step 1: Compute off-diagonal products into r8..r14
    // We compute a[0]*a[1], a[0]*a[2], a[0]*a[3], a[1]*a[2], a[1]*a[3], a[2]*a[3]
    // and accumulate. Then double them and add diagonal squares.

    // --- Off-diagonal: a[0] * a[1..3] ---
    "movq (%[A]), %%rdx\n\t"            // rdx = a[0]
    "mulx 8(%[A]), %%r8, %%r9\n\t"      // r8:r9 = a[0]*a[1]
    "mulx 16(%[A]), %%rax, %%r10\n\t"   // rax:r10 = a[0]*a[2]
    "addq %%rax, %%r9\n\t"
    "mulx 24(%[A]), %%rax, %%r11\n\t"   // rax:r11 = a[0]*a[3]
    "adcq %%rax, %%r10\n\t"
    "adcq $0, %%r11\n\t"

    // --- Off-diagonal: a[1] * a[2..3] ---
    "movq 8(%[A]), %%rdx\n\t"           // rdx = a[1]
    "xorl %%r14d, %%r14d\n\t"           // clear r14 and flags
    "mulx 16(%[A]), %%rax, %%rcx\n\t"   // rax:rcx = a[1]*a[2]
    "adcxq %%rax, %%r10\n\t"
    "adoxq %%rcx, %%r11\n\t"
    "mulx 24(%[A]), %%rax, %%r12\n\t"   // rax:r12 = a[1]*a[3]
    "adcxq %%rax, %%r11\n\t"
    "adoxq %%r14, %%r12\n\t"
    "adcxq %%r14, %%r12\n\t"

    // --- Off-diagonal: a[2] * a[3] ---
    "movq 16(%[A]), %%rdx\n\t"          // rdx = a[2]
    "mulx 24(%[A]), %%rax, %%r13\n\t"   // rax:r13 = a[2]*a[3]
    "addq %%rax, %%r12\n\t"
    "adcq $0, %%r13\n\t"

    // Now r8:r9:r10:r11:r12:r13 = sum of off-diagonal products (positions 1..6)

    // Step 2: Double the off-diagonal sum (shift left by 1)
    "xorl %%r14d, %%r14d\n\t"
    "adcxq %%r8, %%r8\n\t"              // double r8, CF out
    "adcxq %%r9, %%r9\n\t"
    "adcxq %%r10, %%r10\n\t"
    "adcxq %%r11, %%r11\n\t"
    "adcxq %%r12, %%r12\n\t"
    "adcxq %%r13, %%r13\n\t"
    "adcxq %%r14, %%r14\n\t"            // r14 = carry from doubling

    // Step 3: Add diagonal squares a[i]*a[i]
    // a[0]*a[0]
    "movq (%[A]), %%rdx\n\t"
    "mulx %%rdx, %%rax, %%rcx\n\t"      // rax:rcx = a[0]^2
    // rax = r512[0], add rcx to r8
    "movq %%rax, (%[R])\n\t"            // store r512[0] directly (temp use of R)
    "addq %%rcx, %%r8\n\t"              // r512[1] += hi(a[0]^2)

    // a[1]*a[1]
    "movq 8(%[A]), %%rdx\n\t"
    "mulx %%rdx, %%rax, %%rcx\n\t"
    "adcq %%rax, %%r9\n\t"
    "adcq %%rcx, %%r10\n\t"

    // a[2]*a[2]
    "movq 16(%[A]), %%rdx\n\t"
    "mulx %%rdx, %%rax, %%rcx\n\t"
    "adcq %%rax, %%r11\n\t"
    "adcq %%rcx, %%r12\n\t"

    // a[3]*a[3]
    "movq 24(%[A]), %%rdx\n\t"
    "mulx %%rdx, %%rax, %%rcx\n\t"
    "adcq %%rax, %%r13\n\t"
    "adcq %%rcx, %%r14\n\t"

    // Now full 512-bit result:
    // r512[0] = at (%[R]), r512[1..7] = r8:r9:r10:r11 (low) : r12:r13:r14 (high)
    // Reload r512[0] into a register
    "movq (%[R]), %%r15\n\t"            // r15 = r512[0]

    // Step 4: Reduce 512->320: high[4..7] * 0x1000003D1 + low[0..3]
    // high = r12:r13:r14(:0), low = r15:r8:r9:r10:r11 but we only need r15:r8:r9:r10
    // Wait - r512 = r15, r8, r9, r10 (low 256) and r11, r12, r13, r14 (high 256)
    // No! Let me recount:
    // r512[0]=r15, [1]=r8, [2]=r9, [3]=r10, [4]=r11, [5]=r12, [6]=r13, [7]=r14

    "movq $0x1000003D1, %%rdx\n\t"

    "mulx %%r11, %%rax, %%rcx\n\t"      // r11 * K
    "addq %%rax, %%r15\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r11\n\t"

    "mulx %%r12, %%rax, %%rcx\n\t"      // r12 * K
    "addq %%r11, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r8\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r11\n\t"

    "mulx %%r13, %%rax, %%rcx\n\t"      // r13 * K
    "addq %%r11, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r9\n\t"
    "adcq $0, %%rcx\n\t"
    "movq %%rcx, %%r11\n\t"

    "mulx %%r14, %%rax, %%rcx\n\t"      // r14 * K
    "addq %%r11, %%rax\n\t"
    "adcq $0, %%rcx\n\t"
    "addq %%rax, %%r10\n\t"
    "adcq $0, %%rcx\n\t"

    // Step 5: Reduce 320->256: rcx * 0x1000003D1
    "mulx %%rcx, %%rax, %%rcx\n\t"
    "addq %%rax, %%r15\n\t"
    "adcq %%rcx, %%r8\n\t"
    "adcq $0, %%r9\n\t"
    "adcq $0, %%r10\n\t"

    // Store result
    "movq %%r15, (%[R])\n\t"
    "movq %%r8, 8(%[R])\n\t"
    "movq %%r9, 16(%[R])\n\t"
    "movq %%r10, 24(%[R])\n\t"
    "movq $0, 32(%[R])\n\t"

    : /* no output operands */
    : [A] "r" (aa), [R] "r" (rr)
    : "rax", "rcx", "rdx",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "cc", "memory"
  );

#else
  // Fallback: original intrinsics implementation
  unsigned char c;

  uint64_t r512[8];
  uint64_t u10, u11;
  uint64_t t1;
  uint64_t t2;
  uint64_t t[5];


  //k=0
  r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

  //k=1
  t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c,  0,  0, &t1);
  c = _addcarry_u64(0, t[1], t[3], &t[3]);
  c = _addcarry_u64(c, t[4],  0, &t[4]);
  c = _addcarry_u64(c, t1,  0, &t1);
  r512[1] = t[3];

  //k=2
  t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c,  0,  0, &t2);

  u10 = _umul128(a->bits64[1], a->bits64[1], &u11);
  c = _addcarry_u64(0, t[0] , u10, &t[0]);
  c = _addcarry_u64(c, t[1] , u11, &t[1]);
  c = _addcarry_u64(c, t2 ,   0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[2] = t[0];

  //k=3
  t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]);
  u10 = _umul128(a->bits64[1], a->bits64[2], &u11);

  c = _addcarry_u64(0, t[3], u10, &t[3]);
  c = _addcarry_u64(c, t[4], u11, &t[4]);
  c = _addcarry_u64(c,  0,   0, &t1);
  t1 += t1;
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[3] = t[3];

  //k=4
  t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[2], a->bits64[2], &u11);
  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2,  0, &t2);
  r512[4] = t[0];

  //k=5
  t[3] = _umul128(a->bits64[2], a->bits64[3], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1,  0, &t1);
  r512[5] = t[3];

  //k=6
  t[0] = _umul128(a->bits64[3], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  r512[6] = t[0];

  //k=7
  r512[7] = t[1];

  // Reduce from 512 to 320 
  // Reduce from 512 to 320 
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  u10 = _umul128(t[4] + c, 0x1000003D1ULL, &u11);
  c = _addcarry_u64(0, r512[0], u10, bits64 + 0);
  c = _addcarry_u64(c, r512[1], u11, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0, bits64 + 3);
  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#endif

}

static Int _R2o;                               // R^2 for SecpK1 order modular mult
static uint64_t MM64o = 0x4B0DFF665588B13FULL; // 64bits lsb negative inverse of SecpK1 order
static Int *_O;                                // SecpK1 order

void Int::InitK1(Int *order) {
  _O = order;
  _R2o.SetBase16("9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
}

void Int::ModAddK1order(Int *a, Int *b) {
  Add(a,b);
  Sub(_O);
  if (IsNegative())
    Add(_O);
}

void Int::ModMulK1order(Int *a) {

  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 32);
  t.bits64[4] = c;

  for (int i = 1; i < 4; i++) {

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);


  // Normalize

  imm_umul(_R2o.bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 32);
  t.bits64[4] = c;

  for (int i = 1; i < 4; i++) {

    imm_umul(_R2o.bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

}
